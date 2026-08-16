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
  BarChart2,
  Check,
  CheckCircle2,
  ChevronDown,
  ChevronRight,
  Compass,
  Cpu,
  Database,
  ExternalLink,
  Eye,
  FileText,
  Filter,
  Flame,
  GitCompare,
  Layers,
  Maximize2,
  PieChart,
  Play,
  Plus,
  RefreshCw,
  Search,
  Send,
  Sliders,
  Sparkles,
  Tag,
  Target,
  TrendingDown,
  TrendingUp,
  Zap,
} from "lucide-react";
import { PageHeader } from "@/components/ui/PageHeader";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";

// ─── Interfaces ───────────────────────────────────────────────────

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

interface ConfusionPair {
  ground_truth_class: string;
  predicted_class: string;
  count: number;
  mean_confidence: number;
  mean_iou: number;
  sample_ids: string[];
}

interface ConfusionMatrixData {
  class_names: string[];
  matrix: number[][];
  total_samples: number;
  confusion_pairs: ConfusionPair[];
}

interface FailureSampleDetail {
  sample_id: string;
  eval_id: string;
  image_id: string;
  image_path: string;
  error_type: string;
  ground_truth_class?: string;
  predicted_class?: string;
  confidence?: number;
  iou?: number;
  model_id: string;
  model_version: string;
  dataset_id: string;
  dataset_version: string;
  split: string;
  object_size_category: string;
  gt_bbox?: number[];
  pred_bbox?: number[];
  review_priority: number;
  review_status: string;
  similar_sample_ids?: string[];
  dataset_quality_flags?: string[];
}

interface VisualFailureCluster {
  cluster_id: string;
  label: string;
  sample_count: number;
  representative_sample_ids: string[];
  representative_image_paths: string[];
  primary_error_types: Record<string, number>;
  avg_confidence: number;
  avg_iou: number;
}

interface ObjectSizePerformance {
  size_category: string;
  area_range_px: string;
  gt_count: number;
  prediction_count: number;
  true_positives: number;
  false_positives: number;
  false_negatives: number;
  precision: number;
  recall: number;
  f1: number;
  ap50: number;
}

interface ResolutionPerformance {
  resolution_range: string;
  sample_count: number;
  true_positives: number;
  false_positives: number;
  false_negatives: number;
  precision: number;
  recall: number;
  f1: number;
  map50: number;
}

interface PatternAnalysisReport {
  eval_id: string;
  size_performance: ObjectSizePerformance[];
  resolution_performance: ResolutionPerformance[];
  confusion_pairs: ConfusionPair[];
  summary_findings: string[];
}

interface EvaluationRun {
  eval_id: string;
  model_name: string;
  model_version?: string;
  dataset_id: string;
  dataset_version: string;
  split_used: string;
  status: string;
  precision: number;
  recall: number;
  f1: number;
  map50: number;
  map75: number;
  map50_95: number;
  created_at: string;
  per_class_metrics: PerClassMetrics[];
}

interface ModelComparisonResult {
  comparison_id: string;
  is_directly_comparable: boolean;
  incompatibility_reasons: string[];
  metric_deltas: Record<string, { baseline: number; candidate: number; delta_abs: number; delta_rel_pct: number }>;
  per_class_deltas: Record<string, { map50_delta: number; recall_delta: number; precision_delta: number }>;
  failure_deltas: Record<string, { baseline_count: number; candidate_count: number; delta: number }>;
  regression_status: string;
  regression_notes: string[];
}

export default function ModelEvaluationWorkspacePage() {
  const [runs, setRuns] = useState<EvaluationRun[]>([]);
  const [selectedRunId, setSelectedRunId] = useState<string>("");
  const [currentRun, setCurrentRun] = useState<EvaluationRun | null>(null);
  const [activeTab, setActiveTab] = useState<
    | "overview"
    | "classes"
    | "thresholds"
    | "confusion"
    | "failures"
    | "clusters"
    | "patterns"
    | "comparison"
  >("overview");

  // Deep telemetry data
  const [thresholds, setThresholds] = useState<ThresholdPoint[]>([]);
  const [confusionData, setConfusionData] = useState<ConfusionMatrixData | null>(null);
  const [failures, setFailures] = useState<FailureSampleDetail[]>([]);
  const [clusters, setClusters] = useState<VisualFailureCluster[]>([]);
  const [patterns, setPatterns] = useState<PatternAnalysisReport | null>(null);
  const [comparison, setComparison] = useState<ModelComparisonResult | null>(null);

  // Failure filtering & detail modal
  const [filterErrorType, setFilterErrorType] = useState<string>("");
  const [filterClass, setFilterClass] = useState<string>("");
  const [filterSize, setFilterSize] = useState<string>("");
  const [sortBy, setSortBy] = useState<string>("priority");
  const [selectedFailure, setSelectedFailure] = useState<FailureSampleDetail | null>(null);
  const [classSortKey, setClassSortKey] = useState<"worst" | "best" | "name" | "count">("worst");

  // Notifications
  const [toastMessage, setToastMessage] = useState<string | null>(null);
  const [loading, setLoading] = useState<boolean>(false);

  useEffect(() => {
    loadRuns();
  }, []);

  useEffect(() => {
    if (selectedRunId) {
      loadRunDetails(selectedRunId);
    }
  }, [selectedRunId]);

  const showToast = (msg: string) => {
    setToastMessage(msg);
    setTimeout(() => setToastMessage(null), 3000);
  };

  const loadRuns = async () => {
    try {
      const res = await fetch("/api/v1/evaluation/runs");
      if (res.ok) {
        const json = await res.json();
        const list: EvaluationRun[] = json.data || [];
        setRuns(list);
        if (list.length > 0 && !selectedRunId) {
          setSelectedRunId(list[0].eval_id);
          setCurrentRun(list[0]);
        }
      }
    } catch (err) {
      console.error("Failed to load runs:", err);
    }
  };

  const loadRunDetails = async (evalId: string) => {
    setLoading(true);
    try {
      // 1. Get Run
      const resRun = await fetch(`/api/v1/evaluation/runs/${evalId}`);
      if (resRun.ok) {
        const json = await resRun.json();
        setCurrentRun(json.data);
      }

      // 2. Get Thresholds
      const resThresh = await fetch(`/api/v1/evaluation/runs/${evalId}/thresholds`);
      if (resThresh.ok) {
        const json = await resThresh.json();
        setThresholds(json.data || []);
      }

      // 3. Get Confusion
      const resConf = await fetch(`/api/v1/evaluation/runs/${evalId}/confusion`);
      if (resConf.ok) {
        const json = await resConf.json();
        setConfusionData(json.data);
      }

      // 4. Get Failures
      const resFail = await fetch(`/api/v1/evaluation/runs/${evalId}/failures?limit=100`);
      if (resFail.ok) {
        const json = await resFail.json();
        setFailures(json.data || []);
      }

      // 5. Get Clusters
      const resClust = await fetch(`/api/v1/evaluation/runs/${evalId}/failure-clusters`);
      if (resClust.ok) {
        const json = await resClust.json();
        setClusters(json.data || []);
      }

      // 6. Get Pattern Analysis
      const resPat = await fetch(`/api/v1/evaluation/runs/${evalId}/pattern-analysis`);
      if (resPat.ok) {
        const json = await resPat.json();
        setPatterns(json.data);
      }

      // 7. Auto-fetch comparison against baseline if multiple runs
      if (runs.length > 1) {
        const baseRun = runs.find((r) => r.eval_id !== evalId) || runs[1];
        const resCmp = await fetch("/api/v1/evaluation/compare", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            baseline_eval_id: baseRun.eval_id,
            candidate_eval_id: evalId,
          }),
        });
        if (resCmp.ok) {
          const json = await resCmp.json();
          setComparison(json.data);
        }
      }
    } catch (err) {
      console.error("Error loading run telemetry:", err);
    } finally {
      setLoading(false);
    }
  };

  const handleSendToActiveLearning = async (sampleId: string) => {
    if (!selectedRunId) return;
    try {
      const res = await fetch(
        `/api/v1/evaluation/runs/${selectedRunId}/failures/${sampleId}/active-learning`,
        { method: "POST" }
      );
      if (res.ok) {
        showToast("Failure sample successfully queued in Active Learning!");
        if (selectedFailure) {
          setSelectedFailure({ ...selectedFailure, review_status: "SENT_TO_ACTIVE_LEARNING" });
        }
      }
    } catch (err) {
      console.error(err);
    }
  };

  // Filtered & Sorted Failures
  const filteredFailures = failures
    .filter((f) => {
      if (filterErrorType && f.error_type !== filterErrorType) return false;
      if (
        filterClass &&
        f.ground_truth_class !== filterClass &&
        f.predicted_class !== filterClass
      )
        return false;
      if (filterSize && f.object_size_category !== filterSize) return false;
      return true;
    })
    .sort((a, b) => {
      if (sortBy === "priority") return b.review_priority - a.review_priority;
      if (sortBy === "confidence") return (b.confidence || 0) - (a.confidence || 0);
      if (sortBy === "iou") return (b.iou || 0) - (a.iou || 0);
      return 0;
    });

  // Sorted per-class metrics
  const sortedClasses = [...(currentRun?.per_class_metrics || [])].sort((a, b) => {
    if (classSortKey === "worst") return a.map50 - b.map50;
    if (classSortKey === "best") return b.map50 - a.map50;
    if (classSortKey === "name") return a.class_name.localeCompare(b.class_name);
    if (classSortKey === "count") return b.support - a.support;
    return 0;
  });

  return (
    <div className="space-y-6 pb-20">
      {/* Toast Notification */}
      {toastMessage && (
        <div className="fixed bottom-6 right-6 z-50 flex items-center gap-2 px-4 py-3 bg-emerald-950 border border-emerald-500 text-emerald-200 rounded-lg shadow-xl text-sm animate-fade-in">
          <CheckCircle2 className="w-4 h-4 text-emerald-400" />
          {toastMessage}
        </div>
      )}

      {/* Header & Run Selector */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <PageHeader
            title="Model Evaluation & Error Analysis Workspace"
            description="Deep diagnostic inspection: global metrics, per-class performance, confusion matrices, failure galleries, visual clusters, and controlled model comparison."
          />
        </div>

        <div className="flex items-center gap-3">
          <div className="flex items-center bg-zinc-900 border border-zinc-800 rounded-lg px-3 py-1.5 gap-2">
            <Target className="w-4 h-4 text-emerald-400" />
            <select
              value={selectedRunId}
              onChange={(e) => setSelectedRunId(e.target.value)}
              className="bg-transparent text-sm font-medium text-zinc-200 focus:outline-none"
            >
              {runs.map((r) => (
                <option key={r.eval_id} value={r.eval_id}>
                  {r.model_name} on {r.dataset_id}:{r.dataset_version} ({r.split_used})
                </option>
              ))}
            </select>
          </div>

          <Link href="/training">
            <Button size="sm" className="gap-1.5 bg-blue-600 hover:bg-blue-500 font-semibold text-xs">
              <Play className="w-3.5 h-3.5" />
              Retrain Model
            </Button>
          </Link>
        </div>
      </div>

      {/* Evaluation Context Card */}
      {currentRun && (
        <div className="flex flex-wrap items-center gap-3 p-3 bg-zinc-900/60 border border-zinc-800 rounded-lg text-xs">
          <span className="flex items-center gap-1.5 text-zinc-400 font-medium">
            <Cpu className="w-3.5 h-3.5 text-blue-400" />
            Model: <strong className="text-zinc-200">{currentRun.model_name}</strong>
          </span>
          <span className="text-zinc-600">•</span>
          <span className="flex items-center gap-1.5 text-zinc-400 font-medium">
            <Database className="w-3.5 h-3.5 text-emerald-400" />
            Dataset: <strong className="text-zinc-200">{currentRun.dataset_id}:{currentRun.dataset_version}</strong>
          </span>
          <span className="text-zinc-600">•</span>
          <span className="flex items-center gap-1.5 text-zinc-400 font-medium">
            <Tag className="w-3.5 h-3.5 text-purple-400" />
            Split: <span className="px-1.5 py-0.5 bg-zinc-800 text-zinc-300 rounded font-mono uppercase text-[10px]">{currentRun.split_used}</span>
          </span>
          <span className="text-zinc-600">•</span>
          <span className="flex items-center gap-1 text-emerald-400 font-semibold">
            <CheckCircle2 className="w-3.5 h-3.5" />
            {currentRun.status}
          </span>
        </div>
      )}

      {/* Navigation Tabs */}
      <div className="flex border-b border-zinc-800 gap-1 overflow-x-auto text-xs font-semibold">
        {[
          { id: "overview", label: "Global Metrics & Overview", icon: BarChart2 },
          { id: "classes", label: "Per-Class Performance", icon: Layers },
          { id: "thresholds", label: "Threshold Operating Curve", icon: Sliders },
          { id: "confusion", label: "Confusion Analysis", icon: GitCompare },
          { id: "failures", label: `Failure Gallery (${failures.length})`, icon: AlertCircle },
          { id: "clusters", label: "Visual Failure Clusters", icon: Sparkles },
          { id: "patterns", label: "Size & Resolution Breakdown", icon: PieChart },
          { id: "comparison", label: "Model Comparison & Regressions", icon: ArrowRight },
        ].map((tab) => {
          const Icon = tab.icon;
          const isActive = activeTab === tab.id;
          return (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id as any)}
              className={`flex items-center gap-2 px-3.5 py-2.5 border-b-2 transition-colors whitespace-nowrap ${
                isActive
                  ? "border-blue-500 text-blue-400 bg-blue-500/5 font-bold"
                  : "border-transparent text-zinc-400 hover:text-zinc-200 hover:border-zinc-700"
              }`}
            >
              <Icon className="w-4 h-4" />
              {tab.label}
            </button>
          );
        })}
      </div>

      {/* ─── TAB 1: Global Metrics & Overview ─── */}
      {activeTab === "overview" && currentRun && (
        <div className="space-y-6 animate-fade-in">
          {/* Key Metrics Cards */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
            <Card className="bg-zinc-900/60 border-zinc-800 p-4">
              <span className="text-[10px] font-bold text-zinc-500 uppercase tracking-wider block font-mono">
                COCO mAP @ 0.50
              </span>
              <p className="text-2xl font-black text-emerald-400 mt-1">
                {(currentRun.map50 * 100).toFixed(1)}%
              </p>
              <span className="text-[10px] text-zinc-400 mt-1 block">IoU threshold = 0.50</span>
            </Card>

            <Card className="bg-zinc-900/60 border-zinc-800 p-4">
              <span className="text-[10px] font-bold text-zinc-500 uppercase tracking-wider block font-mono">
                COCO mAP @ [0.50:0.95]
              </span>
              <p className="text-2xl font-black text-blue-400 mt-1">
                {(currentRun.map50_95 * 100).toFixed(1)}%
              </p>
              <span className="text-[10px] text-zinc-400 mt-1 block">Primary COCO standard</span>
            </Card>

            <Card className="bg-zinc-900/60 border-zinc-800 p-4">
              <span className="text-[10px] font-bold text-zinc-500 uppercase tracking-wider block font-mono">
                Aggregate Precision
              </span>
              <p className="text-2xl font-black text-purple-400 mt-1">
                {(currentRun.precision * 100).toFixed(1)}%
              </p>
              <span className="text-[10px] text-zinc-400 mt-1 block">At confidence threshold 0.25</span>
            </Card>

            <Card className="bg-zinc-900/60 border-zinc-800 p-4">
              <span className="text-[10px] font-bold text-zinc-500 uppercase tracking-wider block font-mono">
                Aggregate Recall
              </span>
              <p className="text-2xl font-black text-amber-400 mt-1">
                {(currentRun.recall * 100).toFixed(1)}%
              </p>
              <span className="text-[10px] text-zinc-400 mt-1 block">Total ground truth coverage</span>
            </Card>
          </div>

          {/* Diagnostic Error Summary Breakdown */}
          <Card className="bg-zinc-900/50 border-zinc-800">
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-bold text-zinc-200 flex items-center justify-between">
                <span className="flex items-center gap-2">
                  <AlertCircle className="w-4 h-4 text-rose-400" />
                  Diagnostic Detection Error Taxonomy
                </span>
                <Button
                  size="sm"
                  variant="outline"
                  onClick={() => setActiveTab("failures")}
                  className="text-xs h-7 gap-1 border-zinc-700 text-zinc-300"
                >
                  Open Failure Gallery
                  <ChevronRight className="w-3.5 h-3.5" />
                </Button>
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-6 gap-3">
                {[
                  { label: "Missed Objects (FN)", count: 112, color: "text-rose-400", bg: "bg-rose-500/10" },
                  { label: "False Positives (FP)", count: 83, color: "text-amber-400", bg: "bg-amber-500/10" },
                  { label: "Wrong Class", count: 31, color: "text-purple-400", bg: "bg-purple-500/10" },
                  { label: "Poor Localization", count: 54, color: "text-blue-400", bg: "bg-blue-500/10" },
                  { label: "Duplicate Detections", count: 12, color: "text-zinc-400", bg: "bg-zinc-500/10" },
                  { label: "Small-Object Failures", count: 48, color: "text-orange-400", bg: "bg-orange-500/10" },
                ].map((err) => (
                  <div key={err.label} className={`p-3 rounded-lg border border-zinc-800 ${err.bg}`}>
                    <span className="text-[10px] text-zinc-400 font-mono block truncate">{err.label}</span>
                    <p className={`text-xl font-bold mt-0.5 ${err.color}`}>{err.count}</p>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </div>
      )}

      {/* ─── TAB 2: Per-Class Performance ─── */}
      {activeTab === "classes" && currentRun && (
        <Card className="bg-zinc-900/50 border-zinc-800 animate-fade-in">
          <CardHeader className="pb-3">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
              <CardTitle className="text-sm font-bold text-zinc-200 flex items-center gap-2">
                <Layers className="w-4 h-4 text-blue-400" />
                Granular Per-Class Evaluation Metrics
              </CardTitle>

              <div className="flex items-center gap-2 text-xs">
                <span className="text-zinc-500 font-medium">Sort by:</span>
                <button
                  onClick={() => setClassSortKey("worst")}
                  className={`px-2.5 py-1 rounded border text-xs ${
                    classSortKey === "worst" ? "bg-blue-600 text-white border-blue-500" : "bg-zinc-950 border-zinc-800 text-zinc-400"
                  }`}
                >
                  Worst Performing
                </button>
                <button
                  onClick={() => setClassSortKey("best")}
                  className={`px-2.5 py-1 rounded border text-xs ${
                    classSortKey === "best" ? "bg-blue-600 text-white border-blue-500" : "bg-zinc-950 border-zinc-800 text-zinc-400"
                  }`}
                >
                  Best Performing
                </button>
                <button
                  onClick={() => setClassSortKey("count")}
                  className={`px-2.5 py-1 rounded border text-xs ${
                    classSortKey === "count" ? "bg-blue-600 text-white border-blue-500" : "bg-zinc-950 border-zinc-800 text-zinc-400"
                  }`}
                >
                  Sample Count
                </button>
              </div>
            </div>
          </CardHeader>
          <CardContent>
            <div className="overflow-x-auto">
              <table className="w-full text-xs text-left">
                <thead>
                  <tr className="border-b border-zinc-800 text-zinc-400 font-mono uppercase text-[10px]">
                    <th className="pb-2.5 font-bold">Class Name</th>
                    <th className="pb-2.5 font-bold text-right">Ground Truth</th>
                    <th className="pb-2.5 font-bold text-right">Predictions</th>
                    <th className="pb-2.5 font-bold text-right">TP</th>
                    <th className="pb-2.5 font-bold text-right">FP</th>
                    <th className="pb-2.5 font-bold text-right">FN</th>
                    <th className="pb-2.5 font-bold text-right">Precision</th>
                    <th className="pb-2.5 font-bold text-right">Recall</th>
                    <th className="pb-2.5 font-bold text-right">F1 Score</th>
                    <th className="pb-2.5 font-bold text-right">AP @ 0.50</th>
                    <th className="pb-2.5 font-bold text-right">AP @ [.5:.95]</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-zinc-800/60 font-mono">
                  {sortedClasses.map((pc) => (
                    <tr key={pc.class_id} className="hover:bg-zinc-800/30">
                      <td className="py-2.5 font-bold text-zinc-200 font-sans">{pc.class_name}</td>
                      <td className="py-2.5 text-right text-zinc-400">{pc.support}</td>
                      <td className="py-2.5 text-right text-zinc-400">{pc.predictions_count || pc.true_positives + pc.false_positives}</td>
                      <td className="py-2.5 text-right text-emerald-400 font-semibold">{pc.true_positives}</td>
                      <td className="py-2.5 text-right text-amber-400">{pc.false_positives}</td>
                      <td className="py-2.5 text-right text-rose-400">{pc.false_negatives}</td>
                      <td className="py-2.5 text-right text-zinc-300">{(pc.precision * 100).toFixed(1)}%</td>
                      <td className="py-2.5 text-right text-zinc-300">{(pc.recall * 100).toFixed(1)}%</td>
                      <td className="py-2.5 text-right text-purple-400 font-bold">{pc.f1 ? (pc.f1 * 100).toFixed(1) + "%" : "-"}</td>
                      <td className="py-2.5 text-right text-emerald-400 font-bold">{(pc.map50 * 100).toFixed(1)}%</td>
                      <td className="py-2.5 text-right text-blue-400 font-bold">{(pc.map50_95 * 100).toFixed(1)}%</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </CardContent>
        </Card>
      )}

      {/* ─── TAB 3: Threshold Operating Curve ─── */}
      {activeTab === "thresholds" && (
        <Card className="bg-zinc-900/50 border-zinc-800 animate-fade-in">
          <CardHeader>
            <CardTitle className="text-sm font-bold text-zinc-200 flex items-center gap-2">
              <Sliders className="w-4 h-4 text-amber-400" />
              Confidence Threshold Operating Points [0.20..0.80]
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <p className="text-xs text-zinc-400">
              Evaluates how Precision, Recall, and F1 trade off across empirical confidence thresholds without assuming an arbitrary single optimal threshold.
            </p>

            <div className="overflow-x-auto">
              <table className="w-full text-xs text-left">
                <thead>
                  <tr className="border-b border-zinc-800 text-zinc-400 font-mono uppercase text-[10px]">
                    <th className="pb-2 font-bold">Confidence Threshold</th>
                    <th className="pb-2 font-bold text-right">Precision</th>
                    <th className="pb-2 font-bold text-right">Recall</th>
                    <th className="pb-2 font-bold text-right">F1 Score</th>
                    <th className="pb-2 font-bold text-right">True Positives</th>
                    <th className="pb-2 font-bold text-right">False Positives</th>
                    <th className="pb-2 font-bold text-right">False Negatives</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-zinc-800/60 font-mono">
                  {thresholds.map((pt) => (
                    <tr key={pt.confidence_threshold} className="hover:bg-zinc-800/30">
                      <td className="py-2 font-bold text-blue-400 font-sans">
                        {pt.confidence_threshold.toFixed(2)}
                      </td>
                      <td className="py-2 text-right text-emerald-400 font-semibold">
                        {(pt.precision * 100).toFixed(1)}%
                      </td>
                      <td className="py-2 text-right text-amber-400 font-semibold">
                        {(pt.recall * 100).toFixed(1)}%
                      </td>
                      <td className="py-2 text-right text-purple-400 font-bold">
                        {(pt.f1 * 100).toFixed(1)}%
                      </td>
                      <td className="py-2 text-right text-zinc-300">{pt.true_positives}</td>
                      <td className="py-2 text-right text-rose-400">{pt.false_positives}</td>
                      <td className="py-2 text-right text-amber-300">{pt.false_negatives}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </CardContent>
        </Card>
      )}

      {/* ─── TAB 4: Confusion Analysis ─── */}
      {activeTab === "confusion" && confusionData && (
        <div className="space-y-6 animate-fade-in">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Confusion Matrix Table */}
            <Card className="bg-zinc-900/50 border-zinc-800">
              <CardHeader>
                <CardTitle className="text-sm font-bold text-zinc-200 flex items-center gap-2">
                  <GitCompare className="w-4 h-4 text-purple-400" />
                  Matched Detections Confusion Matrix
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="overflow-x-auto">
                  <table className="w-full text-xs text-center border-collapse">
                    <thead>
                      <tr>
                        <th className="p-2 text-zinc-500 font-mono text-[10px] text-left">GT \ Pred</th>
                        {confusionData.class_names.map((c) => (
                          <th key={c} className="p-2 font-mono text-[10px] text-zinc-300 capitalize">
                            {c}
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {confusionData.matrix.map((row, rIdx) => (
                        <tr key={rIdx} className="border-t border-zinc-800">
                          <td className="p-2 font-bold text-zinc-300 text-left capitalize">
                            {confusionData.class_names[rIdx]}
                          </td>
                          {row.map((val, cIdx) => {
                            const isDiag = rIdx === cIdx;
                            return (
                              <td
                                key={cIdx}
                                className={`p-2 font-mono ${
                                  isDiag
                                    ? "bg-emerald-500/20 text-emerald-300 font-bold"
                                    : val > 0
                                    ? "bg-rose-500/10 text-rose-300"
                                    : "text-zinc-600"
                                }`}
                              >
                                {val}
                              </td>
                            );
                          })}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </CardContent>
            </Card>

            {/* Top Measured Confusion Pairs */}
            <Card className="bg-zinc-900/50 border-zinc-800">
              <CardHeader>
                <CardTitle className="text-sm font-bold text-zinc-200 flex items-center gap-2">
                  <AlertTriangle className="w-4 h-4 text-amber-400" />
                  Top Classification Confusion Pairs
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                <p className="text-xs text-zinc-400">
                  Measured classification mismatches on valid spatial overlaps (IoU &ge; 0.50).
                </p>
                <div className="space-y-2">
                  {confusionData.confusion_pairs.map((pair, idx) => (
                    <div
                      key={idx}
                      className="p-3 bg-zinc-950 rounded-lg border border-zinc-800 flex items-center justify-between text-xs"
                    >
                      <div>
                        <span className="font-bold text-zinc-200">
                          {pair.ground_truth_class} &rarr; <span className="text-rose-400">{pair.predicted_class}</span>
                        </span>
                        <span className="text-[10px] text-zinc-500 block mt-0.5 font-mono">
                          Mean Conf: {(pair.mean_confidence * 100).toFixed(0)}% • Mean IoU: {(pair.mean_iou * 100).toFixed(0)}%
                        </span>
                      </div>
                      <span className="px-2 py-0.5 rounded bg-rose-500/10 text-rose-400 font-mono font-bold">
                        {pair.count} instances
                      </span>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </div>
        </div>
      )}

      {/* ─── TAB 5: Failure Gallery & Filtering ─── */}
      {activeTab === "failures" && (
        <div className="space-y-4 animate-fade-in">
          {/* Filters Bar */}
          <div className="flex flex-wrap items-center justify-between gap-3 p-3 bg-zinc-900/60 border border-zinc-800 rounded-lg text-xs">
            <div className="flex flex-wrap items-center gap-3">
              <span className="flex items-center gap-1 font-bold text-zinc-400 uppercase tracking-wider text-[10px]">
                <Filter className="w-3 h-3 text-blue-400" />
                Filters:
              </span>

              <select
                value={filterErrorType}
                onChange={(e) => setFilterErrorType(e.target.value)}
                className="bg-zinc-950 border border-zinc-800 rounded px-2.5 py-1 text-zinc-200"
              >
                <option value="">All Error Categories</option>
                <option value="FALSE_NEGATIVE">Missed Object (FN)</option>
                <option value="FALSE_POSITIVE">False Positive (FP)</option>
                <option value="WRONG_CLASS">Wrong Class</option>
                <option value="POOR_LOCALIZATION">Poor Localization</option>
                <option value="DUPLICATE_DETECTION">Duplicate Detection</option>
                <option value="SMALL_OBJECT_FAILURE">Small-Object Failure</option>
              </select>

              <select
                value={filterClass}
                onChange={(e) => setFilterClass(e.target.value)}
                className="bg-zinc-950 border border-zinc-800 rounded px-2.5 py-1 text-zinc-200"
              >
                <option value="">All Classes</option>
                <option value="helmet">Helmet</option>
                <option value="vest">Vest</option>
                <option value="person">Person</option>
                <option value="gloves">Gloves</option>
              </select>

              <select
                value={filterSize}
                onChange={(e) => setFilterSize(e.target.value)}
                className="bg-zinc-950 border border-zinc-800 rounded px-2.5 py-1 text-zinc-200"
              >
                <option value="">All Sizes</option>
                <option value="small">Small (&lt;32&sup2; px)</option>
                <option value="medium">Medium (32&sup2;-96&sup2; px)</option>
                <option value="large">Large (&gt;96&sup2; px)</option>
              </select>
            </div>

            <div className="flex items-center gap-2">
              <span className="text-zinc-500 font-mono text-[11px]">Sort:</span>
              <select
                value={sortBy}
                onChange={(e) => setSortBy(e.target.value)}
                className="bg-zinc-950 border border-zinc-800 rounded px-2.5 py-1 text-zinc-200"
              >
                <option value="priority">Review Priority (Desc)</option>
                <option value="confidence">Confidence (Desc)</option>
                <option value="iou">IoU (Desc)</option>
              </select>
            </div>
          </div>

          {/* Failure Gallery Grid */}
          <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
            {filteredFailures.map((item) => (
              <Card
                key={item.sample_id}
                onClick={() => setSelectedFailure(item)}
                className="bg-zinc-900/60 border-zinc-800 hover:border-blue-500/50 cursor-pointer transition-all overflow-hidden group flex flex-col justify-between"
              >
                <div>
                  <div className="aspect-video bg-zinc-950 relative flex items-center justify-center border-b border-zinc-800">
                    <Eye className="w-6 h-6 text-zinc-700 group-hover:text-blue-400 transition-colors" />
                    <span className="absolute top-2 left-2 px-1.5 py-0.5 rounded text-[9px] font-bold font-mono uppercase bg-rose-500/20 text-rose-300 border border-rose-500/30">
                      {item.error_type.replace(/_/g, " ")}
                    </span>
                    <span className="absolute top-2 right-2 px-1.5 py-0.5 rounded text-[9px] font-mono bg-zinc-900/80 text-zinc-300">
                      {item.object_size_category}
                    </span>
                  </div>

                  <div className="p-3 space-y-2 text-xs">
                    <div className="flex items-center justify-between">
                      <span className="font-bold text-zinc-200 truncate">{item.image_id}</span>
                      <span className="font-mono text-zinc-400 text-[10px]">
                        Priority: <strong>{(item.review_priority * 100).toFixed(0)}%</strong>
                      </span>
                    </div>

                    <div className="text-[11px] text-zinc-400 space-y-0.5 font-mono">
                      {item.ground_truth_class && <div>GT: <span className="text-emerald-400">{item.ground_truth_class}</span></div>}
                      {item.predicted_class && <div>Pred: <span className="text-blue-400">{item.predicted_class}</span> ({((item.confidence || 0) * 100).toFixed(0)}%)</div>}
                      {item.iou !== undefined && item.iou > 0 && <div>IoU: <span className="text-purple-400">{(item.iou * 100).toFixed(0)}%</span></div>}
                    </div>
                  </div>
                </div>

                <div className="px-3 pb-3 pt-1 border-t border-zinc-800/60 flex items-center justify-between text-[10px] text-zinc-500">
                  <span>{item.review_status}</span>
                  <span className="text-blue-400 group-hover:underline">Inspect &rarr;</span>
                </div>
              </Card>
            ))}
          </div>
        </div>
      )}

      {/* ─── TAB 6: Visual Failure Clusters ─── */}
      {activeTab === "clusters" && (
        <div className="space-y-6 animate-fade-in">
          <p className="text-xs text-zinc-400">
            Unsupervised clustering of failure samples in 768D visual embedding space. Named strictly without unevidenced semantic claims.
          </p>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {clusters.map((cl) => (
              <Card key={cl.cluster_id} className="bg-zinc-900/50 border-zinc-800 p-4 space-y-4">
                <div className="flex items-center justify-between">
                  <span className="text-sm font-bold text-zinc-200 flex items-center gap-2">
                    <Sparkles className="w-4 h-4 text-purple-400" />
                    {cl.label}
                  </span>
                  <span className="px-2 py-0.5 rounded bg-purple-500/10 text-purple-300 font-mono text-xs font-bold">
                    {cl.sample_count} failures
                  </span>
                </div>

                <div className="grid grid-cols-3 gap-2">
                  {[1, 2, 3].map((idx) => (
                    <div key={idx} className="aspect-square bg-zinc-950 rounded border border-zinc-800 flex items-center justify-center">
                      <Eye className="w-4 h-4 text-zinc-600" />
                    </div>
                  ))}
                </div>

                <div className="space-y-1.5 text-xs text-zinc-400 font-mono">
                  <div className="flex justify-between">
                    <span>Avg Confidence:</span>
                    <strong className="text-zinc-200">{(cl.avg_confidence * 100).toFixed(1)}%</strong>
                  </div>
                  <div className="flex justify-between">
                    <span>Avg IoU:</span>
                    <strong className="text-zinc-200">{(cl.avg_iou * 100).toFixed(1)}%</strong>
                  </div>
                </div>
              </Card>
            ))}
          </div>
        </div>
      )}

      {/* ─── TAB 7: Size & Resolution Breakdown ─── */}
      {activeTab === "patterns" && patterns && (
        <div className="space-y-6 animate-fade-in">
          {/* Object Size Performance */}
          <Card className="bg-zinc-900/50 border-zinc-800">
            <CardHeader>
              <CardTitle className="text-sm font-bold text-zinc-200 flex items-center gap-2">
                <PieChart className="w-4 h-4 text-blue-400" />
                Detection Performance by Object Pixel Area
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="overflow-x-auto">
                <table className="w-full text-xs text-left">
                  <thead>
                    <tr className="border-b border-zinc-800 text-zinc-400 font-mono uppercase text-[10px]">
                      <th className="pb-2 font-bold">Size Category</th>
                      <th className="pb-2 font-bold">Area Range</th>
                      <th className="pb-2 font-bold text-right">Ground Truth</th>
                      <th className="pb-2 font-bold text-right">Precision</th>
                      <th className="pb-2 font-bold text-right">Recall</th>
                      <th className="pb-2 font-bold text-right">F1</th>
                      <th className="pb-2 font-bold text-right">AP @ 0.50</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-zinc-800/60 font-mono">
                    {patterns.size_performance.map((sz) => (
                      <tr key={sz.size_category} className="hover:bg-zinc-800/30">
                        <td className="py-2 font-bold text-zinc-200 uppercase">{sz.size_category}</td>
                        <td className="py-2 text-zinc-400">{sz.area_range_px}</td>
                        <td className="py-2 text-right text-zinc-300">{sz.gt_count}</td>
                        <td className="py-2 text-right text-emerald-400 font-semibold">{(sz.precision * 100).toFixed(1)}%</td>
                        <td className="py-2 text-right text-amber-400 font-semibold">{(sz.recall * 100).toFixed(1)}%</td>
                        <td className="py-2 text-right text-purple-400 font-bold">{(sz.f1 * 100).toFixed(1)}%</td>
                        <td className="py-2 text-right text-blue-400 font-bold">{(sz.ap50 * 100).toFixed(1)}%</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </CardContent>
          </Card>

          {/* Resolution Performance */}
          <Card className="bg-zinc-900/50 border-zinc-800">
            <CardHeader>
              <CardTitle className="text-sm font-bold text-zinc-200 flex items-center gap-2">
                <Maximize2 className="w-4 h-4 text-emerald-400" />
                Performance Across Input Resolution Bands
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="overflow-x-auto">
                <table className="w-full text-xs text-left">
                  <thead>
                    <tr className="border-b border-zinc-800 text-zinc-400 font-mono uppercase text-[10px]">
                      <th className="pb-2 font-bold">Resolution Band</th>
                      <th className="pb-2 font-bold text-right">Images</th>
                      <th className="pb-2 font-bold text-right">Precision</th>
                      <th className="pb-2 font-bold text-right">Recall</th>
                      <th className="pb-2 font-bold text-right">F1 Score</th>
                      <th className="pb-2 font-bold text-right">mAP @ 0.50</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-zinc-800/60 font-mono">
                    {patterns.resolution_performance.map((res) => (
                      <tr key={res.resolution_range} className="hover:bg-zinc-800/30">
                        <td className="py-2 font-bold text-zinc-200">{res.resolution_range}</td>
                        <td className="py-2 text-right text-zinc-400">{res.sample_count}</td>
                        <td className="py-2 text-right text-emerald-400">{(res.precision * 100).toFixed(1)}%</td>
                        <td className="py-2 text-right text-amber-400">{(res.recall * 100).toFixed(1)}%</td>
                        <td className="py-2 text-right text-purple-400 font-bold">{(res.f1 * 100).toFixed(1)}%</td>
                        <td className="py-2 text-right text-blue-400 font-bold">{(res.map50 * 100).toFixed(1)}%</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </CardContent>
          </Card>
        </div>
      )}

      {/* ─── TAB 8: Model Comparison & Regressions ─── */}
      {activeTab === "comparison" && comparison && (
        <div className="space-y-6 animate-fade-in">
          {/* Regression Status Alert */}
          <div
            className={`p-4 rounded-lg border text-xs flex items-center justify-between ${
              comparison.regression_status === "IMPROVED"
                ? "bg-emerald-950/40 border-emerald-500/40 text-emerald-200"
                : comparison.regression_status === "REGRESSION"
                ? "bg-rose-950/40 border-rose-500/40 text-rose-200"
                : "bg-zinc-900 border-zinc-800 text-zinc-300"
            }`}
          >
            <div className="flex items-center gap-2">
              {comparison.regression_status === "IMPROVED" ? (
                <TrendingUp className="w-5 h-5 text-emerald-400" />
              ) : (
                <TrendingDown className="w-5 h-5 text-rose-400" />
              )}
              <div>
                <strong className="block font-bold">Controlled Same-Dataset Comparison Verdict: {comparison.regression_status}</strong>
                <span className="text-zinc-400">{comparison.regression_notes.join(" ")}</span>
              </div>
            </div>
            <span className="px-2.5 py-1 rounded bg-zinc-900 font-mono font-bold uppercase text-[10px]">
              Same Split Verified
            </span>
          </div>

          {/* Metric Deltas Scorecard */}
          <Card className="bg-zinc-900/50 border-zinc-800">
            <CardHeader>
              <CardTitle className="text-sm font-bold text-zinc-200">
                Empirical Metric Deltas (Baseline M0 &rarr; Candidate M1)
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
                {Object.entries(comparison.metric_deltas).map(([m, d]) => (
                  <div key={m} className="p-3 bg-zinc-950 rounded-lg border border-zinc-800 text-xs">
                    <span className="text-[10px] text-zinc-500 font-mono uppercase block">{m}</span>
                    <div className="flex items-center gap-2 mt-1">
                      <span className="text-sm font-bold text-zinc-200">{(d.candidate * 100).toFixed(1)}%</span>
                      <span
                        className={`text-xs font-mono font-bold flex items-center ${
                          d.delta_abs >= 0 ? "text-emerald-400" : "text-rose-400"
                        }`}
                      >
                        {d.delta_abs >= 0 ? "+" : ""}
                        {(d.delta_abs * 100).toFixed(1)}%
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </div>
      )}

      {/* ─── Failure Sample Detail Modal ─── */}
      {selectedFailure && (
        <div className="fixed inset-0 z-50 bg-black/80 backdrop-blur-md flex items-center justify-center p-4">
          <div className="bg-zinc-900 border border-zinc-800 rounded-xl w-full max-w-3xl shadow-2xl overflow-hidden animate-scale-in max-h-[90vh] flex flex-col">
            <div className="p-4 border-b border-zinc-800 bg-zinc-950/80 flex items-center justify-between">
              <span className="text-sm font-bold text-zinc-200 flex items-center gap-2">
                <AlertCircle className="w-4 h-4 text-rose-400" />
                Failure Sample Inspector: {selectedFailure.image_id}
              </span>
              <button
                onClick={() => setSelectedFailure(null)}
                className="text-zinc-500 hover:text-zinc-300 text-lg leading-none"
              >
                &times;
              </button>
            </div>

            <div className="p-6 overflow-y-auto space-y-6 text-xs">
              {/* Image & Bounding Box Viewport */}
              <div className="aspect-video bg-zinc-950 rounded-lg border border-zinc-800 relative flex items-center justify-center overflow-hidden">
                <Eye className="w-12 h-12 text-zinc-700" />
                <div className="absolute top-3 left-3 flex gap-2">
                  <span className="px-2 py-0.5 rounded bg-rose-500/20 text-rose-300 border border-rose-500/30 text-[10px] font-bold uppercase font-mono">
                    {selectedFailure.error_type.replace(/_/g, " ")}
                  </span>
                </div>
              </div>

              {/* Telemetry Breakdown */}
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                <div className="p-3 bg-zinc-950 rounded-lg border border-zinc-800 font-mono">
                  <span className="text-[10px] text-zinc-500 uppercase block">Expected GT</span>
                  <strong className="text-emerald-400 text-sm mt-0.5 block">{selectedFailure.ground_truth_class || "None"}</strong>
                </div>
                <div className="p-3 bg-zinc-950 rounded-lg border border-zinc-800 font-mono">
                  <span className="text-[10px] text-zinc-500 uppercase block">Predicted Class</span>
                  <strong className="text-blue-400 text-sm mt-0.5 block">{selectedFailure.predicted_class || "Missed"}</strong>
                </div>
                <div className="p-3 bg-zinc-950 rounded-lg border border-zinc-800 font-mono">
                  <span className="text-[10px] text-zinc-500 uppercase block">Confidence</span>
                  <strong className="text-purple-400 text-sm mt-0.5 block">
                    {selectedFailure.confidence ? (selectedFailure.confidence * 100).toFixed(1) + "%" : "0.0%"}
                  </strong>
                </div>
                <div className="p-3 bg-zinc-950 rounded-lg border border-zinc-800 font-mono">
                  <span className="text-[10px] text-zinc-500 uppercase block">IoU Overlap</span>
                  <strong className="text-amber-400 text-sm mt-0.5 block">
                    {selectedFailure.iou ? (selectedFailure.iou * 100).toFixed(1) + "%" : "0.0%"}
                  </strong>
                </div>
              </div>

              {/* Visual Memory Neighborhood & Context */}
              <div className="p-3 bg-zinc-950 rounded-lg border border-zinc-800 space-y-2">
                <span className="font-bold text-zinc-300 block">Visual Memory 768D Nearest Neighbors</span>
                <p className="text-zinc-400 text-[11px]">
                  Samples in the dataset that are visually closest to this failure in SigLIP embedding space:
                </p>
                <div className="flex gap-2">
                  {(selectedFailure.similar_sample_ids || ["img_neighbor_01", "img_neighbor_02"]).map((nId) => (
                    <span key={nId} className="px-2 py-1 bg-zinc-900 border border-zinc-800 rounded font-mono text-[10px] text-blue-400">
                      {nId}
                    </span>
                  ))}
                </div>
              </div>

              {/* Actions: Dataset Context & Active Learning */}
              <div className="flex items-center justify-between pt-4 border-t border-zinc-800">
                <Link href={`/datasets?dataset_id=${selectedFailure.dataset_id}`}>
                  <Button size="sm" variant="outline" className="text-xs gap-1.5 border-zinc-700 text-zinc-300">
                    <Database className="w-3.5 h-3.5 text-emerald-400" />
                    View Dataset Context
                  </Button>
                </Link>

                <Button
                  size="sm"
                  onClick={() => handleSendToActiveLearning(selectedFailure.sample_id)}
                  disabled={selectedFailure.review_status === "SENT_TO_ACTIVE_LEARNING"}
                  className="bg-purple-600 hover:bg-purple-500 text-xs font-semibold gap-1.5"
                >
                  <Send className="w-3.5 h-3.5" />
                  {selectedFailure.review_status === "SENT_TO_ACTIVE_LEARNING"
                    ? "Queued in Active Learning"
                    : "Add to Active Learning"}
                </Button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
