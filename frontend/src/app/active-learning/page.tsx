"use client";

import React, { useEffect, useState } from "react";
import Link from "next/link";
import {
  Activity,
  AlertCircle,
  AlertTriangle,
  ArrowRight,
  BarChart2,
  CheckCircle2,
  ChevronRight,
  Compass,
  Database,
  Eye,
  Filter,
  Flame,
  HelpCircle,
  Info,
  Layers,
  ListFilter,
  Plus,
  RefreshCw,
  Search,
  ShieldAlert,
  ShieldCheck,
  Sparkles,
  Sliders,
  Tag,
  Target,
  ThumbsDown,
  ThumbsUp,
  XCircle,
} from "lucide-react";
import { PageHeader } from "@/components/ui/PageHeader";
import { Button } from "@/components/ui/Button";

// ─── Interfaces ───────────────────────────────────────────────────

type SelectionStrategy = "UNCERTAINTY" | "DIVERSITY" | "UNCERTAINTY_DIVERSITY" | "NOVELTY";
type ReviewStatus = "UNREVIEWED" | "ACCEPTED" | "REJECTED" | "SKIPPED" | "MARKED_FOR_LABELING";

interface SignalWeights {
  uncertainty: number;
  novelty: number;
  diversity: number;
  failure: number;
  quality: number;
}

interface SampleSignals {
  image_id: string;
  image_path: string;
  uncertainty_score: number;
  novelty_score: number;
  diversity_score: number;
  failure_score: number;
  quality_score: number;
  composite_score: number;
}

interface RankedSample {
  rank: number;
  image_id: string;
  image_path: string;
  composite_score: number;
  signals: SampleSignals;
  recommendation_reason: string;
  review_status: ReviewStatus;
  notes?: string;
}

interface ActiveLearningRun {
  run_id: string;
  experiment_id?: string;
  model_id: string;
  dataset_id: string;
  candidate_pool_id: string;
  strategy: SelectionStrategy;
  weights: SignalWeights;
  top_k: number;
  selected_samples: RankedSample[];
  status: string;
  created_at: string;
}

interface SelectionBiasReport {
  run_id: string;
  strategy: SelectionStrategy;
  total_selected: number;
  class_distribution: Record<string, number>;
  quality_distribution: Record<string, number>;
  confidence_distribution: Record<string, number>;
  bias_summary: string;
}

interface StrategyComparisonResult {
  dataset_id: string;
  model_id: string;
  strategy_a: SelectionStrategy;
  strategy_b: SelectionStrategy;
  overlap_count: number;
  unique_a_count: number;
  unique_b_count: number;
  diversity_delta: number;
  uncertainty_delta: number;
  summary_notes: string;
}

interface MetricDelta {
  baseline_val: number;
  retrained_val: number;
  delta: number;
  percent_change: number;
}

interface ActiveLearningIteration {
  iteration_id: string;
  baseline_dataset_id: string;
  baseline_model_id: string;
  baseline_evaluation_id: string;
  active_learning_run_id: string;
  reviewed_samples_count: number;
  new_dataset_version: string;
  retrained_run_id: string;
  retrained_model_id: string;
  retrained_evaluation_id: string;
  map50_delta: MetricDelta;
  map50_95_delta: MetricDelta;
  precision_delta: MetricDelta;
  recall_delta: MetricDelta;
  verdict: "IMPROVED" | "REGRESSED" | "NEUTRAL";
  verdict_summary: string;
  created_at: string;
}

export default function ActiveLearningPage() {
  // Navigation Tab State
  const [activeTab, setActiveTab] = useState<"studio" | "queue" | "bias" | "compare" | "loop">("studio");

  // Loop Execution State
  const [iteration, setIteration] = useState<ActiveLearningIteration | null>(null);
  const [executingLoop, setExecutingLoop] = useState<boolean>(false);

  // Selection Generator State
  const [datasetId, setDatasetId] = useState<string>("safety_v2");
  const [modelId, setModelId] = useState<string>("yolo11s.pt");
  const [candidatePool, setCandidatePool] = useState<string>("unlabeled_pool_v2");
  const [strategy, setStrategy] = useState<SelectionStrategy>("UNCERTAINTY_DIVERSITY");
  const [topK, setTopK] = useState<number>(25);

  // Weights State
  const [weights, setWeights] = useState<SignalWeights>({
    uncertainty: 0.40,
    novelty: 0.25,
    diversity: 0.25,
    failure: 0.10,
    quality: 0.00,
  });

  // Current Run & Telemetry State
  const [currentRun, setCurrentRun] = useState<ActiveLearningRun | null>(null);
  const [biasReport, setBiasReport] = useState<SelectionBiasReport | null>(null);
  const [comparisonResult, setComparisonResult] = useState<StrategyComparisonResult | null>(null);
  const [loading, setLoading] = useState<boolean>(false);

  // Strategy Comparison Form State
  const [compareStratA, setCompareStratA] = useState<SelectionStrategy>("UNCERTAINTY");
  const [compareStratB, setCompareStratB] = useState<SelectionStrategy>("UNCERTAINTY_DIVERSITY");
  const [comparing, setComparing] = useState<boolean>(false);

  // Selected Sample Detail Drawer State
  const [inspectSample, setInspectSample] = useState<RankedSample | null>(null);

  useEffect(() => {
    // Generate initial run on mount
    handleGenerateRecommendations();
  }, []);

  const handleGenerateRecommendations = async () => {
    setLoading(true);
    try {
      const res = await fetch("/api/v1/active-learning/runs", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          dataset_id: datasetId,
          model_id: modelId,
          strategy: strategy,
          weights: weights,
          top_k: topK,
        }),
      });

      if (res.ok) {
        const data: ActiveLearningRun = await res.json();
        setCurrentRun(data);

        // Fetch Bias Report for the run
        const biasRes = await fetch(`/api/v1/active-learning/runs/${data.run_id}/bias`);
        if (biasRes.ok) {
          setBiasReport(await biasRes.json());
        }
      }
    } catch (err) {
      console.error("Failed to generate active learning recommendations:", err);
    } finally {
      setLoading(false);
    }
  };

  const handleReviewDecision = async (
    imageId: string,
    reviewStatus: ReviewStatus,
    notes?: string
  ) => {
    if (!currentRun) return;

    try {
      const res = await fetch("/api/v1/active-learning/review", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          run_id: currentRun.run_id,
          image_id: imageId,
          status: reviewStatus,
          notes: notes,
        }),
      });

      if (res.ok) {
        const updatedRun: ActiveLearningRun = await res.json();
        setCurrentRun(updatedRun);
        if (inspectSample?.image_id === imageId) {
          const updatedSample = updatedRun.selected_samples.find((s) => s.image_id === imageId);
          if (updatedSample) setInspectSample(updatedSample);
        }
      }
    } catch (err) {
      console.error("Failed to submit review decision:", err);
    }
  };

  const handleCompareStrategies = async (e: React.FormEvent) => {
    e.preventDefault();
    setComparing(true);
    try {
      const res = await fetch("/api/v1/active-learning/compare", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          dataset_id: datasetId,
          model_id: modelId,
          strategy_a: compareStratA,
          strategy_b: compareStratB,
          top_k: topK,
        }),
      });

      if (res.ok) {
        setComparisonResult(await res.json());
      }
    } catch (err) {
      console.error("Failed to compare strategies:", err);
    } finally {
      setComparing(false);
    }
  };

  const handleExecuteLoop = async () => {
    if (!currentRun) return;
    setExecutingLoop(true);
    try {
      const res = await fetch("/api/v1/active-learning/loop", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          baseline_dataset_id: datasetId,
          baseline_model_id: modelId,
          active_learning_run_id: currentRun.run_id,
          new_version_tag: "v2.1",
        }),
      });

      if (res.ok) {
        const data: ActiveLearningIteration = await res.json();
        setIteration(data);
        setActiveTab("loop");
      }
    } catch (err) {
      console.error("Failed to execute retraining loop:", err);
    } finally {
      setExecutingLoop(false);
    }
  };

  return (
    <div className="flex flex-col min-h-screen bg-[#0a0a0a] text-neutral-200 font-inter">
      {/* Page Header */}
      <PageHeader
        title="Active Learning & Sample Selection Studio"
        description="Intelligent multi-signal sample recommendation engine answering: 'Which images should we label or inspect next?'"
        breadcrumbs={["VisionForge", "Active Learning"]}
        actions={
          <div className="flex items-center gap-2">
            <Button
              variant="secondary"
              icon={<BarChart2 className="w-4 h-4 text-purple-400" />}
              onClick={() => setActiveTab("compare")}
            >
              Compare Strategies
            </Button>
            <Button
              variant="primary"
              icon={<Sparkles className="w-4 h-4" />}
              onClick={handleGenerateRecommendations}
              disabled={loading}
            >
              {loading ? "Ranking..." : "Generate Recommendations"}
            </Button>
          </div>
        }
      />

      <div className="p-6 space-y-6 flex-1">
        {/* Top Tab Navigation */}
        <div className="flex items-center bg-[#141414] border border-white/10 rounded-xl p-1 w-fit flex-wrap gap-1">
          <button
            onClick={() => setActiveTab("studio")}
            className={`px-4 py-2 rounded-lg text-xs font-medium transition-all ${
              activeTab === "studio"
                ? "bg-blue-600/20 text-blue-400 border border-blue-500/30 shadow"
                : "text-neutral-400 hover:text-white"
            }`}
          >
            Recommendation Studio
          </button>
          <button
            onClick={() => setActiveTab("queue")}
            className={`px-4 py-2 rounded-lg text-xs font-medium transition-all flex items-center gap-2 ${
              activeTab === "queue"
                ? "bg-blue-600/20 text-blue-400 border border-blue-500/30 shadow"
                : "text-neutral-400 hover:text-white"
            }`}
          >
            Human Review Queue
            {currentRun && (
              <span className="text-[10px] bg-blue-500/20 text-blue-400 px-1.5 py-0.5 rounded font-mono">
                {currentRun.selected_samples.length}
              </span>
            )}
          </button>
          <button
            onClick={() => setActiveTab("bias")}
            className={`px-4 py-2 rounded-lg text-xs font-medium transition-all ${
              activeTab === "bias"
                ? "bg-blue-600/20 text-blue-400 border border-blue-500/30 shadow"
                : "text-neutral-400 hover:text-white"
            }`}
          >
            Selection Bias Telemetry
          </button>
          <button
            onClick={() => setActiveTab("compare")}
            className={`px-4 py-2 rounded-lg text-xs font-medium transition-all ${
              activeTab === "compare"
                ? "bg-blue-600/20 text-blue-400 border border-blue-500/30 shadow"
                : "text-neutral-400 hover:text-white"
            }`}
          >
            Strategy Comparison
          </button>
          <button
            onClick={() => setActiveTab("loop")}
            className={`px-4 py-2 rounded-lg text-xs font-medium transition-all flex items-center gap-1.5 ${
              activeTab === "loop"
                ? "bg-emerald-600/20 text-emerald-400 border border-emerald-500/30 shadow"
                : "text-neutral-400 hover:text-white"
            }`}
          >
            <Activity className="w-3.5 h-3.5 text-emerald-400" />
            Retraining & Performance Verdict
          </button>
        </div>

        {/* ─── TAB 1: RECOMMENDATION STUDIO ────────────────────────────────── */}
        {activeTab === "studio" && (
          <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
            {/* Left Control Panel */}
            <div className="lg:col-span-1 space-y-6">
              {/* Configuration Panel */}
              <div className="bg-[#121212] border border-white/10 rounded-xl p-5 space-y-4 text-xs">
                <h3 className="text-xs font-semibold uppercase tracking-wider text-neutral-400 flex items-center gap-2">
                  <Sliders className="w-4 h-4 text-blue-400" />
                  Selection Parameters
                </h3>

                <div>
                  <label className="text-neutral-400 block mb-1 font-medium">Target Model</label>
                  <select
                    value={modelId}
                    onChange={(e) => setModelId(e.target.value)}
                    className="w-full bg-[#1a1a1a] border border-white/10 rounded-lg px-3 py-2 text-white font-mono"
                  >
                    <option value="yolo11s.pt">YOLO11s Safety (v1.0)</option>
                    <option value="rtdetr_l.pt">RT-DETR-L Safety (v1.0)</option>
                  </select>
                </div>

                <div>
                  <label className="text-neutral-400 block mb-1 font-medium">Dataset Context</label>
                  <select
                    value={datasetId}
                    onChange={(e) => setDatasetId(e.target.value)}
                    className="w-full bg-[#1a1a1a] border border-white/10 rounded-lg px-3 py-2 text-white font-mono"
                  >
                    <option value="safety_v2">Safety Dataset v2 (v2.0)</option>
                    <option value="construction_v1">Construction Equipment v1</option>
                  </select>
                </div>

                <div>
                  <label className="text-neutral-400 block mb-1 font-medium">Candidate Image Pool</label>
                  <select
                    value={candidatePool}
                    onChange={(e) => setCandidatePool(e.target.value)}
                    className="w-full bg-[#1a1a1a] border border-white/10 rounded-lg px-3 py-2 text-white font-mono"
                  >
                    <option value="unlabeled_pool_v2">Unlabeled Images Pool (500 images)</option>
                    <option value="validation_candidates">Validation Candidates Pool</option>
                    <option value="new_ingestion_batch">New Camera Feed Ingestion Batch</option>
                  </select>
                </div>

                {/* Test Set Protection Banner */}
                <div className="bg-emerald-950/30 border border-emerald-500/30 rounded-lg p-3 space-y-1">
                  <div className="flex items-center gap-1.5 text-emerald-400 font-semibold">
                    <ShieldCheck className="w-4 h-4 shrink-0" />
                    <span>Test-Set Protection Active</span>
                  </div>
                  <p className="text-[10px] text-neutral-400">
                    Evaluation test split images are strictly excluded from candidate selection.
                  </p>
                </div>

                {/* Strategy Selector */}
                <div className="space-y-2 pt-2 border-t border-white/10">
                  <label className="text-neutral-400 block font-medium">Selection Strategy</label>
                  <div className="space-y-1.5">
                    {[
                      {
                        id: "UNCERTAINTY_DIVERSITY",
                        label: "Uncertainty + Diversity",
                        desc: "Combines prediction ambiguity with visual coverage distance.",
                      },
                      {
                        id: "UNCERTAINTY",
                        label: "Uncertainty Sampling",
                        desc: "Ranks candidates by prediction confidence ambiguity.",
                      },
                      {
                        id: "DIVERSITY",
                        label: "Diversity Sampling",
                        desc: "Farthest-Point Greedy k-Center embedding sampling.",
                      },
                      {
                        id: "NOVELTY",
                        label: "Novelty Sampling",
                        desc: "Ranks candidates by distance from dataset centroid.",
                      },
                    ].map((s) => (
                      <label
                        key={s.id}
                        className={`flex items-start gap-2.5 p-2.5 rounded-lg border cursor-pointer transition-all ${
                          strategy === s.id
                            ? "bg-blue-600/15 border-blue-500/50 text-white"
                            : "bg-[#181818] border-white/5 text-neutral-400 hover:text-white"
                        }`}
                      >
                        <input
                          type="radio"
                          name="strategy"
                          value={s.id}
                          checked={strategy === s.id}
                          onChange={() => setStrategy(s.id as SelectionStrategy)}
                          className="mt-0.5"
                        />
                        <div>
                          <div className="font-semibold text-xs text-white">{s.label}</div>
                          <div className="text-[10px] text-neutral-500">{s.desc}</div>
                        </div>
                      </label>
                    ))}
                  </div>
                </div>

                {/* Top-K Selection */}
                <div>
                  <label className="text-neutral-400 block mb-1 font-medium">
                    Sample Batch Size (Top-K)
                  </label>
                  <div className="grid grid-cols-4 gap-1.5 font-mono">
                    {[10, 25, 50, 100].map((k) => (
                      <button
                        key={k}
                        type="button"
                        onClick={() => setTopK(k)}
                        className={`py-1.5 rounded text-xs border font-semibold ${
                          topK === k
                            ? "bg-blue-600/20 border-blue-500/50 text-blue-400"
                            : "bg-[#181818] border-white/10 text-neutral-400 hover:text-white"
                        }`}
                      >
                        {k}
                      </button>
                    ))}
                  </div>
                </div>
              </div>
            </div>

            {/* Right Recommended Candidates Area */}
            <div className="lg:col-span-3 space-y-6">
              {/* Active Run Banner */}
              {currentRun && (
                <div className="bg-[#121212] border border-white/10 rounded-xl p-4 flex flex-wrap items-center justify-between gap-4">
                  <div className="space-y-1">
                    <div className="flex items-center gap-2">
                      <span className="text-sm font-semibold text-white">
                        Run: {currentRun.run_id}
                      </span>
                      <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-blue-500/15 text-blue-400 border border-blue-500/30">
                        {currentRun.strategy}
                      </span>
                    </div>
                    <div className="text-xs text-neutral-400 font-mono">
                      Generated {currentRun.selected_samples.length} top candidates for model &apos;
                      {currentRun.model_id}&apos; on dataset &apos;{currentRun.dataset_id}&apos;.
                    </div>
                  </div>

                  <Button
                    variant="secondary"
                    size="sm"
                    icon={<Eye className="w-3.5 h-3.5 text-blue-400" />}
                    onClick={() => setActiveTab("queue")}
                  >
                    Open Review Queue
                  </Button>
                </div>
              )}

              {/* Candidates Grid */}
              <div className="bg-[#121212] border border-white/10 rounded-xl overflow-hidden">
                <div className="p-4 border-b border-white/10 flex justify-between items-center bg-[#161616]">
                  <h3 className="text-xs font-semibold uppercase tracking-wider text-neutral-400 flex items-center gap-2">
                    <Sparkles className="w-4 h-4 text-blue-400" />
                    Recommended Candidate Samples ({currentRun?.selected_samples.length || 0})
                  </h3>
                  <span className="text-[11px] text-neutral-500 font-mono">
                    Ranked by Multi-Signal Composite Score
                  </span>
                </div>

                <div className="p-4 grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
                  {currentRun?.selected_samples.map((sample) => (
                    <div
                      key={sample.image_id}
                      className="bg-[#181818] border border-white/10 hover:border-blue-500/40 rounded-xl overflow-hidden p-4 space-y-3 flex flex-col justify-between transition-all"
                    >
                      {/* Top Header */}
                      <div className="flex justify-between items-start">
                        <div className="flex items-center gap-2">
                          <span className="text-xs font-bold font-mono px-2 py-0.5 rounded bg-blue-600/20 text-blue-400 border border-blue-500/30">
                            #{sample.rank}
                          </span>
                          <span className="text-xs font-mono text-neutral-400 truncate max-w-[120px]">
                            {sample.image_id}
                          </span>
                        </div>
                        <span className="text-xs font-bold font-mono text-emerald-400">
                          Score: {sample.composite_score.toFixed(2)}
                        </span>
                      </div>

                      {/* Mock Image Box */}
                      <div className="h-32 bg-[#0c0c0c] border border-white/5 rounded-lg flex flex-col items-center justify-center p-2 text-center relative group">
                        <div className="text-[11px] text-neutral-500 font-mono truncate max-w-full">
                          {sample.image_path.split("/").pop()}
                        </div>
                        <div className="text-[10px] text-neutral-600 mt-1">
                          Candidate Image Preview
                        </div>

                        {/* Hover Overlay triggers */}
                        <div className="absolute inset-0 bg-black/80 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center gap-2 p-2">
                          <Link href="/search">
                            <Button variant="secondary" size="sm" className="text-[10px] py-1 px-2">
                              Find Similar
                            </Button>
                          </Link>
                          <Link href="/explorer">
                            <Button variant="secondary" size="sm" className="text-[10px] py-1 px-2">
                              Explorer
                            </Button>
                          </Link>
                        </div>
                      </div>

                      {/* Recommendation Reason */}
                      <p className="text-xs text-neutral-300 line-clamp-2 leading-relaxed bg-[#121212] p-2 rounded border border-white/5 font-mono text-[11px]">
                        {sample.recommendation_reason}
                      </p>

                      {/* Signal Breakdown Progress Bars */}
                      <div className="space-y-1.5 text-[10px] font-mono">
                        <div className="flex justify-between text-neutral-400">
                          <span>Uncertainty:</span>
                          <span className="text-blue-400 font-bold">
                            {(sample.signals.uncertainty_score * 100).toFixed(0)}%
                          </span>
                        </div>
                        <div className="w-full bg-[#121212] h-1.5 rounded-full overflow-hidden">
                          <div
                            className="bg-blue-500 h-full"
                            style={{ width: `${sample.signals.uncertainty_score * 100}%` }}
                          />
                        </div>

                        <div className="flex justify-between text-neutral-400">
                          <span>Novelty:</span>
                          <span className="text-purple-400 font-bold">
                            {(sample.signals.novelty_score * 100).toFixed(0)}%
                          </span>
                        </div>
                        <div className="w-full bg-[#121212] h-1.5 rounded-full overflow-hidden">
                          <div
                            className="bg-purple-500 h-full"
                            style={{ width: `${sample.signals.novelty_score * 100}%` }}
                          />
                        </div>
                      </div>

                      {/* Human Review Quick Actions */}
                      <div className="pt-2 border-t border-white/5 flex items-center justify-between gap-1">
                        <span
                          className={`text-[9px] font-mono px-1.5 py-0.5 rounded uppercase ${
                            sample.review_status === "ACCEPTED"
                              ? "bg-emerald-500/20 text-emerald-400 border border-emerald-500/30"
                              : sample.review_status === "REJECTED"
                              ? "bg-red-500/20 text-red-400 border border-red-500/30"
                              : sample.review_status === "MARKED_FOR_LABELING"
                              ? "bg-purple-500/20 text-purple-400 border border-purple-500/30"
                              : "bg-neutral-800 text-neutral-400"
                          }`}
                        >
                          {sample.review_status}
                        </span>

                        <div className="flex items-center gap-1">
                          <button
                            onClick={() => handleReviewDecision(sample.image_id, "ACCEPTED")}
                            className="p-1 rounded bg-emerald-500/10 text-emerald-400 hover:bg-emerald-500/20 border border-emerald-500/30"
                            title="Accept Sample"
                          >
                            <ThumbsUp className="w-3.5 h-3.5" />
                          </button>
                          <button
                            onClick={() => handleReviewDecision(sample.image_id, "REJECTED")}
                            className="p-1 rounded bg-red-500/10 text-red-400 hover:bg-red-500/20 border border-red-500/30"
                            title="Reject Sample"
                          >
                            <ThumbsDown className="w-3.5 h-3.5" />
                          </button>
                          <button
                            onClick={() =>
                              handleReviewDecision(sample.image_id, "MARKED_FOR_LABELING")
                            }
                            className="p-1 rounded bg-purple-500/10 text-purple-400 hover:bg-purple-500/20 border border-purple-500/30"
                            title="Mark for Labeling"
                          >
                            <Tag className="w-3.5 h-3.5" />
                          </button>
                        </div>
                      </div>
                    </div>
                  ))}

                  {loading && (
                    <div className="col-span-full py-16 text-center text-xs text-neutral-500 font-mono space-y-2">
                      <RefreshCw className="w-8 h-8 mx-auto text-blue-500 animate-spin" />
                      <div>Computing multi-signal ranking and farthest-point diversity...</div>
                    </div>
                  )}
                </div>
              </div>
            </div>
          </div>
        )}

        {/* ─── TAB 2: HUMAN REVIEW QUEUE ──────────────────────────────────── */}
        {activeTab === "queue" && (
          <div className="bg-[#121212] border border-white/10 rounded-xl p-6 space-y-6">
            <div className="flex justify-between items-center border-b border-white/10 pb-3">
              <h3 className="text-xs font-semibold uppercase tracking-wider text-neutral-400 flex items-center gap-2">
                <Eye className="w-4 h-4 text-blue-400" />
                Human Review Queue ({currentRun?.selected_samples.length || 0} Candidates)
              </h3>
              <span className="text-xs text-neutral-500 font-mono">
                Human-in-the-Loop Review: Selected samples require explicit review before labeling
              </span>
            </div>

            <div className="space-y-4">
              {currentRun?.selected_samples.map((sample) => (
                <div
                  key={sample.image_id}
                  className="bg-[#161616] border border-white/10 rounded-xl p-5 flex flex-col md:flex-row md:items-center justify-between gap-6"
                >
                  <div className="space-y-2 flex-1">
                    <div className="flex items-center gap-3">
                      <span className="text-sm font-bold font-mono px-2 py-0.5 rounded bg-blue-600/20 text-blue-400 border border-blue-500/30">
                        Rank #{sample.rank}
                      </span>
                      <span className="text-xs font-mono text-white font-semibold">
                        {sample.image_id}
                      </span>
                      <span className="text-xs font-mono text-emerald-400">
                        Score: {sample.composite_score.toFixed(2)}
                      </span>
                    </div>

                    <p className="text-xs text-neutral-300 font-mono">{sample.recommendation_reason}</p>

                    <div className="flex flex-wrap gap-4 text-xs font-mono text-neutral-400 pt-1">
                      <span>Uncertainty: {(sample.signals.uncertainty_score * 100).toFixed(0)}%</span>
                      <span>Novelty: {(sample.signals.novelty_score * 100).toFixed(0)}%</span>
                      <span>Diversity: {(sample.signals.diversity_score * 100).toFixed(0)}%</span>
                      <span>Quality: {(sample.signals.quality_score * 100).toFixed(0)}%</span>
                    </div>
                  </div>

                  {/* Actions */}
                  <div className="flex items-center gap-2 shrink-0">
                    <Button
                      variant={sample.review_status === "ACCEPTED" ? "primary" : "secondary"}
                      size="sm"
                      onClick={() => handleReviewDecision(sample.image_id, "ACCEPTED")}
                    >
                      Accept
                    </Button>
                    <Button
                      variant={sample.review_status === "REJECTED" ? "primary" : "secondary"}
                      size="sm"
                      onClick={() => handleReviewDecision(sample.image_id, "REJECTED")}
                    >
                      Reject
                    </Button>
                    <Button
                      variant={sample.review_status === "MARKED_FOR_LABELING" ? "primary" : "secondary"}
                      size="sm"
                      onClick={() => handleReviewDecision(sample.image_id, "MARKED_FOR_LABELING")}
                    >
                      Mark for Labeling
                    </Button>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* ─── TAB 3: SELECTION BIAS TELEMETRY ─────────────────────────────── */}
        {activeTab === "bias" && (
          <div className="bg-[#121212] border border-white/10 rounded-xl p-6 space-y-6">
            <h3 className="text-xs font-semibold uppercase tracking-wider text-neutral-400 flex items-center gap-2">
              <BarChart2 className="w-4 h-4 text-purple-400" />
              Selection Bias Analysis & Distribution Telemetry
            </h3>

            {biasReport && (
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6 text-xs font-mono">
                <div className="bg-[#181818] border border-white/5 rounded-xl p-4 space-y-3">
                  <h4 className="font-semibold text-white">Predicted Class Distribution</h4>
                  {Object.entries(biasReport.class_distribution).map(([cls, count]) => (
                    <div key={cls} className="flex justify-between py-1 border-b border-white/5">
                      <span className="text-neutral-400">{cls}:</span>
                      <span className="text-blue-400 font-bold">{count} samples</span>
                    </div>
                  ))}
                </div>

                <div className="bg-[#181818] border border-white/5 rounded-xl p-4 space-y-3">
                  <h4 className="font-semibold text-white">Image Quality Breakdown</h4>
                  {Object.entries(biasReport.quality_distribution).map(([q, count]) => (
                    <div key={q} className="flex justify-between py-1 border-b border-white/5">
                      <span className="text-neutral-400 uppercase">{q} Quality:</span>
                      <span className="text-emerald-400 font-bold">{count} samples</span>
                    </div>
                  ))}
                </div>

                <div className="bg-[#181818] border border-white/5 rounded-xl p-4 space-y-3">
                  <h4 className="font-semibold text-white">Confidence Quartile Statistics</h4>
                  {Object.entries(biasReport.confidence_distribution).map(([q, val]) => (
                    <div key={q} className="flex justify-between py-1 border-b border-white/5">
                      <span className="text-neutral-400 uppercase">{q}:</span>
                      <span className="text-purple-400 font-bold">{val}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}

        {/* ─── TAB 4: STRATEGY COMPARISON ──────────────────────────────────── */}
        {activeTab === "compare" && (
          <div className="bg-[#121212] border border-white/10 rounded-xl p-6 space-y-6">
            <h3 className="text-xs font-semibold uppercase tracking-wider text-neutral-400 flex items-center gap-2">
              <Compass className="w-4 h-4 text-emerald-400" />
              Comparative Active Learning Strategy Analysis
            </h3>

            <form onSubmit={handleCompareStrategies} className="grid grid-cols-1 md:grid-cols-3 gap-4 text-xs">
              <div>
                <label className="text-neutral-400 block mb-1">Strategy A</label>
                <select
                  value={compareStratA}
                  onChange={(e) => setCompareStratA(e.target.value as SelectionStrategy)}
                  className="w-full bg-[#1a1a1a] border border-white/10 rounded-lg p-2 text-white"
                >
                  <option value="UNCERTAINTY">Uncertainty Sampling</option>
                  <option value="DIVERSITY">Diversity Sampling</option>
                  <option value="NOVELTY">Novelty Sampling</option>
                </select>
              </div>

              <div>
                <label className="text-neutral-400 block mb-1">Strategy B</label>
                <select
                  value={compareStratB}
                  onChange={(e) => setCompareStratB(e.target.value as SelectionStrategy)}
                  className="w-full bg-[#1a1a1a] border border-white/10 rounded-lg p-2 text-white"
                >
                  <option value="UNCERTAINTY_DIVERSITY">Uncertainty + Diversity</option>
                  <option value="DIVERSITY">Diversity Sampling</option>
                  <option value="NOVELTY">Novelty Sampling</option>
                </select>
              </div>

              <div className="flex items-end">
                <Button variant="primary" size="sm" className="w-full" disabled={comparing}>
                  {comparing ? "Comparing..." : "Run Comparative Analysis"}
                </Button>
              </div>
            </form>

            {comparisonResult && (
              <div className="bg-[#181818] border border-white/5 rounded-xl p-5 space-y-4 text-xs font-mono">
                <div className="text-blue-400 font-semibold">{comparisonResult.summary_notes}</div>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 pt-2">
                  <div className="bg-[#121212] p-3 rounded">
                    <div className="text-neutral-400">Overlapping Samples</div>
                    <div className="text-lg font-bold text-white">{comparisonResult.overlap_count}</div>
                  </div>
                  <div className="bg-[#121212] p-3 rounded">
                    <div className="text-neutral-400">Unique to Strategy A</div>
                    <div className="text-lg font-bold text-blue-400">{comparisonResult.unique_a_count}</div>
                  </div>
                  <div className="bg-[#121212] p-3 rounded">
                    <div className="text-neutral-400">Unique to Strategy B</div>
                    <div className="text-lg font-bold text-purple-400">{comparisonResult.unique_b_count}</div>
                  </div>
                  <div className="bg-[#121212] p-3 rounded">
                    <div className="text-neutral-400">Diversity Delta</div>
                    <div className="text-lg font-bold text-emerald-400">{comparisonResult.diversity_delta}</div>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}

        {/* ─── TAB 5: CLOSED-LOOP RETRAINING & PERFORMANCE VERDICT ───────────── */}
        {activeTab === "loop" && (
          <div className="bg-[#121212] border border-white/10 rounded-xl p-6 space-y-6">
            <div className="flex flex-wrap justify-between items-center border-b border-white/10 pb-4 gap-4">
              <div>
                <h3 className="text-xs font-semibold uppercase tracking-wider text-neutral-400 flex items-center gap-2">
                  <Activity className="w-4 h-4 text-emerald-400" />
                  Closed-Loop Retraining & Performance Improvement Verdict
                </h3>
                <p className="text-xs text-neutral-400 mt-1">
                  Executes controlled model retraining with reviewed active learning samples and measures empirical accuracy delta on the untouched test split.
                </p>
              </div>

              <Button
                variant="primary"
                size="sm"
                icon={<RefreshCw className={`w-3.5 h-3.5 ${executingLoop ? "animate-spin" : ""}`} />}
                onClick={handleExecuteLoop}
                disabled={executingLoop || !currentRun}
              >
                {executingLoop ? "Retraining & Evaluating..." : "Execute Retraining Loop & Measure Delta"}
              </Button>
            </div>

            {/* Visual Flowchart Diagram */}
            <div className="bg-[#161616] border border-white/10 rounded-xl p-5 space-y-3">
              <div className="text-[11px] font-semibold text-neutral-400 uppercase tracking-wider">
                Active Learning Closed-Loop Execution Flowchart
              </div>
              <div className="flex flex-wrap items-center justify-center gap-2 py-4 font-mono text-[11px] text-neutral-300 bg-[#0c0c0c] border border-white/5 rounded-lg">
                <span className="bg-blue-600/20 text-blue-400 border border-blue-500/30 px-2.5 py-1 rounded">
                  Baseline Dataset (D0)
                </span>
                <span className="text-neutral-500">→</span>
                <span className="bg-purple-600/20 text-purple-400 border border-purple-500/30 px-2.5 py-1 rounded">
                  Train (M0)
                </span>
                <span className="text-neutral-500">→</span>
                <span className="bg-cyan-600/20 text-cyan-400 border border-cyan-500/30 px-2.5 py-1 rounded">
                  Evaluate (E0)
                </span>
                <span className="text-neutral-500">→</span>
                <span className="bg-amber-600/20 text-amber-400 border border-amber-500/30 px-2.5 py-1 rounded">
                  Active Learning & Review
                </span>
                <span className="text-neutral-500">→</span>
                <span className="bg-indigo-600/20 text-indigo-400 border border-indigo-500/30 px-2.5 py-1 rounded">
                  New Version (D1)
                </span>
                <span className="text-neutral-500">→</span>
                <span className="bg-emerald-600/20 text-emerald-400 border border-emerald-500/30 font-bold px-2.5 py-1 rounded">
                  Verdict (E1)
                </span>
              </div>
            </div>

            {/* Empirical Performance Telemetry Result */}
            {iteration ? (
              <div className="space-y-6">
                {/* Verdict Header Banner */}
                <div className="bg-[#181818] border border-emerald-500/30 rounded-xl p-5 flex flex-wrap items-center justify-between gap-4">
                  <div className="space-y-1">
                    <div className="flex items-center gap-3">
                      <span className="text-xs font-bold uppercase tracking-wider px-3 py-1 rounded bg-emerald-500/20 text-emerald-400 border border-emerald-500/40 font-mono">
                        VERDICT: {iteration.verdict}
                      </span>
                      <span className="text-xs text-neutral-400 font-mono">
                        Iteration ID: {iteration.iteration_id}
                      </span>
                    </div>
                    <p className="text-xs text-neutral-300 font-mono pt-1">
                      {iteration.verdict_summary}
                    </p>
                  </div>

                  <div className="text-right font-mono text-xs text-neutral-400">
                    <div>Dataset Bump: <span className="text-white font-bold">{iteration.baseline_dataset_id} → {iteration.new_dataset_version}</span></div>
                    <div>Active Candidates Added: <span className="text-emerald-400 font-bold">{iteration.reviewed_samples_count} accepted</span></div>
                  </div>
                </div>

                {/* Metric Delta Comparison Table */}
                <div className="bg-[#161616] border border-white/10 rounded-xl overflow-hidden text-xs font-mono">
                  <div className="p-3 bg-[#1c1c1c] border-b border-white/10 font-semibold uppercase text-neutral-400 tracking-wider">
                    Empirical Accuracy Delta (Untouched Test Split Evaluation)
                  </div>
                  <div className="divide-y divide-white/5">
                    <div className="grid grid-cols-4 p-3 font-semibold text-neutral-400 bg-[#141414]">
                      <div>Metric</div>
                      <div>Baseline Model (M0)</div>
                      <div>Retrained Model (M1)</div>
                      <div>Performance Delta (Δ)</div>
                    </div>

                    <div className="grid grid-cols-4 p-3 text-neutral-200 hover:bg-white/5 transition-colors">
                      <div className="font-bold text-white">mAP@50</div>
                      <div>{iteration.map50_delta.baseline_val.toFixed(4)}</div>
                      <div>{iteration.map50_delta.retrained_val.toFixed(4)}</div>
                      <div className="text-emerald-400 font-bold">
                        +{iteration.map50_delta.delta.toFixed(4)} (+{iteration.map50_delta.percent_change}%)
                      </div>
                    </div>

                    <div className="grid grid-cols-4 p-3 text-neutral-200 hover:bg-white/5 transition-colors">
                      <div className="font-bold text-white">mAP@50:95</div>
                      <div>{iteration.map50_95_delta.baseline_val.toFixed(4)}</div>
                      <div>{iteration.map50_95_delta.retrained_val.toFixed(4)}</div>
                      <div className="text-emerald-400 font-bold">
                        +{iteration.map50_95_delta.delta.toFixed(4)} (+{iteration.map50_95_delta.percent_change}%)
                      </div>
                    </div>

                    <div className="grid grid-cols-4 p-3 text-neutral-200 hover:bg-white/5 transition-colors">
                      <div className="font-bold text-white">Precision</div>
                      <div>{iteration.precision_delta.baseline_val.toFixed(4)}</div>
                      <div>{iteration.precision_delta.retrained_val.toFixed(4)}</div>
                      <div className="text-emerald-400 font-bold">
                        +{iteration.precision_delta.delta.toFixed(4)} (+{iteration.precision_delta.percent_change}%)
                      </div>
                    </div>

                    <div className="grid grid-cols-4 p-3 text-neutral-200 hover:bg-white/5 transition-colors">
                      <div className="font-bold text-white">Recall</div>
                      <div>{iteration.recall_delta.baseline_val.toFixed(4)}</div>
                      <div>{iteration.recall_delta.retrained_val.toFixed(4)}</div>
                      <div className="text-emerald-400 font-bold">
                        +{iteration.recall_delta.delta.toFixed(4)} (+{iteration.recall_delta.percent_change}%)
                      </div>
                    </div>
                  </div>
                </div>

                <div className="bg-emerald-950/20 border border-emerald-500/20 rounded-xl p-4 flex items-center justify-between text-xs font-mono text-emerald-400">
                  <div className="flex items-center gap-2">
                    <ShieldCheck className="w-4 h-4 shrink-0" />
                    <span>Scientific Verification: Retrained evaluation performed on identical, untouched evaluation test split.</span>
                  </div>
                  <span className="text-neutral-400 text-[10px]">Test Set Immutability Guaranteed</span>
                </div>
              </div>
            ) : (
              <div className="bg-[#161616] border border-white/5 rounded-xl p-12 text-center text-xs text-neutral-500 font-mono space-y-3">
                <Activity className="w-8 h-8 mx-auto text-emerald-500" />
                <div>Click &apos;Execute Retraining Loop &amp; Measure Delta&apos; to evaluate performance improvement after active learning.</div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
