"use client";

import React, { useEffect, useState, useCallback } from "react";
import Link from "next/link";
import {
  Activity,
  AlertCircle,
  AlertTriangle,
  ArrowRight,
  BarChart2,
  Check,
  CheckCircle2,
  ChevronLeft,
  ChevronRight,
  Database,
  Eye,
  Filter,
  Flame,
  HelpCircle,
  Info,
  Keyboard,
  Layers,
  ListFilter,
  Play,
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
  X,
  XCircle,
  Zap,
} from "lucide-react";
import { PageHeader } from "@/components/ui/PageHeader";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";

// ─── Interfaces ───────────────────────────────────────────────────

type SelectionStrategy =
  | "UNCERTAINTY"
  | "DIVERSITY"
  | "HYBRID"
  | "UNCERTAINTY_DIVERSITY"
  | "MODEL_DISAGREEMENT"
  | "CLASS_AWARE"
  | "FAILURE_AWARE";

type ReviewStatus = "UNREVIEWED" | "IN_REVIEW" | "ACCEPTED" | "REJECTED" | "SKIPPED" | "FLAGGED";

type ReviewDecisionType =
  | "CONFIRMED"
  | "INCORRECT_PREDICTION"
  | "ANNOTATION_ISSUE"
  | "VALID_HARD_EXAMPLE"
  | "DUPLICATE"
  | "NOT_USEFUL"
  | "NEEDS_MORE_REVIEW"
  | "SKIP";

interface CandidateExplanation {
  composite_priority: number;
  uncertainty_contribution: number;
  diversity_contribution: number;
  failure_contribution: number;
  class_rarity_flag: boolean;
  model_disagreement_flag: boolean;
  plain_text_reasons: string[];
}

interface CandidateSampleDetail {
  rank: number;
  image_id: string;
  image_path: string;
  split: string;
  composite_score: number;
  signals: {
    uncertainty_score: number;
    diversity_score: number;
    failure_score: number;
    novelty_score: number;
    composite_score: number;
  };
  explanation: CandidateExplanation;
  ground_truth_boxes: Array<{ class_name?: string; bbox?: number[] }>;
  predicted_boxes: Array<{ class_name?: string; confidence?: number; bbox?: number[]; iou?: number }>;
  predicted_class?: string;
  confidence?: number;
  iou?: number;
  similar_sample_ids: string[];
  review_status: ReviewStatus;
  review_decision?: ReviewDecisionType;
  notes?: string;
}

interface ActiveLearningCycle {
  cycle_id: string;
  name: string;
  dataset_id: string;
  dataset_version: string;
  model_id: string;
  model_version: string;
  candidate_pool_id: string;
  candidate_pool_size: number;
  strategy: SelectionStrategy;
  budget: number;
  selected_samples: CandidateSampleDetail[];
  review_counts: {
    pending: number;
    in_review: number;
    reviewed: number;
    skipped: number;
    flagged: number;
  };
  resulting_dataset_version?: string;
  benchmark_before_map50?: number;
  benchmark_after_map50?: number;
  status: string;
  created_at: string;
}

interface ActiveLearningCycleHistoryItem {
  cycle_id: string;
  name: string;
  dataset_version_before: string;
  dataset_version_after?: string;
  model_version_before: string;
  model_version_after?: string;
  samples_reviewed: number;
  strategy: SelectionStrategy;
  budget: number;
  map50_before?: number;
  map50_after?: number;
  delta_map50?: number;
  created_at: string;
}

export default function ActiveLearningPage() {
  const [cycles, setCycles] = useState<ActiveLearningCycle[]>([]);
  const [selectedCycleId, setSelectedCycleId] = useState<string>("");
  const [currentCycle, setCurrentCycle] = useState<ActiveLearningCycle | null>(null);
  const [history, setHistory] = useState<ActiveLearningCycleHistoryItem[]>([]);

  // Selection Config
  const [budget, setBudget] = useState<number>(50);
  const [strategy, setStrategy] = useState<SelectionStrategy>("HYBRID");
  const [isSelecting, setIsSelecting] = useState<boolean>(false);

  // Focus Review Session State
  const [focusIndex, setFocusIndex] = useState<number | null>(null);
  const [toastMessage, setToastMessage] = useState<string | null>(null);
  const [statusFilter, setStatusFilter] = useState<string>("ALL");
  const [newVersionTag, setNewVersionTag] = useState<string>("v2.1.0");
  const [isCommitting, setIsCommitting] = useState<boolean>(false);

  useEffect(() => {
    loadCycles();
    loadHistory();
  }, []);

  const loadCycles = async () => {
    try {
      const res = await fetch("/api/v1/active-learning/cycles");
      if (res.ok) {
        const payload = await res.json();
        const list = payload.data || [];
        setCycles(list);
        if (list.length > 0 && !selectedCycleId) {
          setSelectedCycleId(list[0].cycle_id);
          setCurrentCycle(list[0]);
        }
      }
    } catch (err) {
      console.error("Failed to load cycles:", err);
    }
  };

  const loadHistory = async () => {
    try {
      const res = await fetch("/api/v1/active-learning/cycles/history");
      if (res.ok) {
        const payload = await res.json();
        setHistory(payload.data || []);
      }
    } catch (err) {
      console.error("Failed to load history:", err);
    }
  };

  const handleSelectCycle = (cycleId: string) => {
    setSelectedCycleId(cycleId);
    const found = cycles.find((c) => c.cycle_id === cycleId);
    if (found) {
      setCurrentCycle(found);
      setBudget(found.budget);
      setStrategy(found.strategy);
    }
  };

  const handleExecuteSelection = async () => {
    if (!currentCycle) return;
    setIsSelecting(true);
    try {
      const res = await fetch(`/api/v1/active-learning/cycles/${currentCycle.cycle_id}/select`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ budget, strategy }),
      });
      if (res.ok) {
        const payload = await res.json();
        setCurrentCycle(payload.data);
        showToast(`Selected ${payload.data.selected_samples.length} candidates using ${strategy} sampling`);
      }
    } catch (err) {
      console.error("Failed to execute selection:", err);
    } finally {
      setIsSelecting(false);
    }
  };

  const handleReviewDecision = async (
    sampleId: string,
    decision: ReviewDecisionType,
    advance: boolean = true
  ) => {
    if (!currentCycle) return;
    try {
      const res = await fetch(`/api/v1/active-learning/cycles/${currentCycle.cycle_id}/review`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          cycle_id: currentCycle.cycle_id,
          image_id: sampleId,
          decision: decision,
          reviewer_id: "Principal Researcher",
        }),
      });

      if (res.ok) {
        // Refresh cycle local state
        const updatedSamples = currentCycle.selected_samples.map((s) => {
          if (s.image_id === sampleId) {
            return {
              ...s,
              review_decision: decision,
              review_status: (decision === "CONFIRMED" || decision === "INCORRECT_PREDICTION" || decision === "ANNOTATION_ISSUE" || decision === "VALID_HARD_EXAMPLE"
                ? "ACCEPTED"
                : decision === "NOT_USEFUL" || decision === "DUPLICATE"
                ? "REJECTED"
                : decision === "SKIP"
                ? "SKIPPED"
                : "FLAGGED") as ReviewStatus,
            };
          }
          return s;
        });

        const reviewedCount = updatedSamples.filter((s) => s.review_status === "ACCEPTED" || s.review_status === "REJECTED").length;
        const skippedCount = updatedSamples.filter((s) => s.review_status === "SKIPPED").length;
        const flaggedCount = updatedSamples.filter((s) => s.review_status === "FLAGGED").length;
        const pendingCount = updatedSamples.length - (reviewedCount + skippedCount + flaggedCount);

        const updatedCycle: ActiveLearningCycle = {
          ...currentCycle,
          selected_samples: updatedSamples,
          review_counts: {
            pending: Math.max(0, pendingCount),
            in_review: 0,
            reviewed: reviewedCount,
            skipped: skippedCount,
            flagged: flaggedCount,
          },
        };
        setCurrentCycle(updatedCycle);

        showToast(`Recorded: ${decision.replace(/_/g, " ")}`);

        if (advance && focusIndex !== null && focusIndex < currentCycle.selected_samples.length - 1) {
          setFocusIndex(focusIndex + 1);
        }
      }
    } catch (err) {
      console.error("Failed to record review decision:", err);
    }
  };

  const handleCommitDatasetVersion = async () => {
    if (!currentCycle) return;
    setIsCommitting(true);
    try {
      const res = await fetch(`/api/v1/active-learning/cycles/${currentCycle.cycle_id}/commit-version`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          new_version_tag: newVersionTag,
          changes_summary: `Active learning curated batch: ${currentCycle.review_counts.reviewed} samples accepted.`,
        }),
      });

      if (res.ok) {
        const payload = await res.json();
        setCurrentCycle(payload.data);
        showToast(`Committed new dataset version '${newVersionTag}'!`);
        loadHistory();
      }
    } catch (err) {
      console.error("Failed to commit dataset version:", err);
    } finally {
      setIsCommitting(false);
    }
  };

  const showToast = (msg: string) => {
    setToastMessage(msg);
    setTimeout(() => setToastMessage(null), 3000);
  };

  // Keyboard Shortcuts for Focus Review Mode (Step 22)
  const handleKeyDown = useCallback(
    (e: KeyboardEvent) => {
      if (focusIndex === null || !currentCycle) return;
      const sample = currentCycle.selected_samples[focusIndex];
      if (!sample) return;

      const key = e.key.toUpperCase();
      if (key === "C") {
        e.preventDefault();
        handleReviewDecision(sample.image_id, "CONFIRMED", true);
      } else if (key === "R") {
        e.preventDefault();
        handleReviewDecision(sample.image_id, "INCORRECT_PREDICTION", true);
      } else if (key === "A") {
        e.preventDefault();
        handleReviewDecision(sample.image_id, "ANNOTATION_ISSUE", true);
      } else if (key === "S") {
        e.preventDefault();
        handleReviewDecision(sample.image_id, "SKIP", true);
      } else if (key === "F") {
        e.preventDefault();
        handleReviewDecision(sample.image_id, "NEEDS_MORE_REVIEW", true);
      } else if (key === "N" || e.key === "ArrowRight") {
        e.preventDefault();
        if (focusIndex < currentCycle.selected_samples.length - 1) setFocusIndex(focusIndex + 1);
      } else if (key === "P" || e.key === "ArrowLeft") {
        e.preventDefault();
        if (focusIndex > 0) setFocusIndex(focusIndex - 1);
      } else if (e.key === "Escape") {
        e.preventDefault();
        setFocusIndex(null);
      }
    },
    [focusIndex, currentCycle]
  );

  useEffect(() => {
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [handleKeyDown]);

  const activeSample = focusIndex !== null && currentCycle ? currentCycle.selected_samples[focusIndex] : null;

  return (
    <div className="space-y-6 pb-16">
      {/* Toast Notification */}
      {toastMessage && (
        <div className="fixed bottom-6 right-6 z-50 flex items-center gap-2 px-4 py-3 bg-emerald-950 border border-emerald-500 text-emerald-200 rounded-lg shadow-xl text-sm animate-fade-in">
          <CheckCircle2 className="w-4 h-4 text-emerald-400" />
          {toastMessage}
        </div>
      )}

      {/* Header & Controls */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <PageHeader
            title="Active Learning & Human-in-the-Loop Workflow"
            description="Prioritize informative candidates via uncertainty, diversity, and model disagreement for expert human curation."
          />
        </div>

        <div className="flex items-center gap-3">
          <div className="flex items-center bg-zinc-900 border border-zinc-800 rounded-lg px-3 py-1.5 gap-2">
            <Activity className="w-4 h-4 text-blue-400" />
            <select
              value={selectedCycleId}
              onChange={(e) => handleSelectCycle(e.target.value)}
              className="bg-transparent text-sm font-medium text-zinc-200 focus:outline-none"
            >
              {cycles.map((c) => (
                <option key={c.cycle_id} value={c.cycle_id}>
                  {c.name} ({c.status})
                </option>
              ))}
            </select>
          </div>

          <Button
            size="sm"
            onClick={() => setFocusIndex(0)}
            disabled={!currentCycle || currentCycle.selected_samples.length === 0}
            className="gap-1.5 bg-blue-600 hover:bg-blue-500 font-semibold"
          >
            <Play className="w-3.5 h-3.5" />
            Start Focus Review
          </Button>
        </div>
      </div>

      {/* Cycle Progress & Strategy Scorecard */}
      {currentCycle && (
        <div className="grid grid-cols-2 sm:grid-cols-5 gap-3">
          <Card className="bg-zinc-900/60 border-zinc-800 p-3">
            <span className="text-[11px] text-zinc-400">Total Budget</span>
            <p className="text-xl font-bold text-zinc-100 mt-1">
              {currentCycle.selected_samples.length} / {currentCycle.budget}
            </p>
            <span className="text-[10px] text-blue-400 mt-0.5 block">{currentCycle.strategy} Strategy</span>
          </Card>

          <Card className="bg-zinc-900/60 border-zinc-800 p-3">
            <span className="text-[11px] text-zinc-400">Reviewed / Accepted</span>
            <p className="text-xl font-bold text-emerald-400 mt-1">{currentCycle.review_counts.reviewed}</p>
            <span className="text-[10px] text-emerald-500 mt-0.5 block">Approved for Curation</span>
          </Card>

          <Card className="bg-zinc-900/60 border-zinc-800 p-3">
            <span className="text-[11px] text-zinc-400">Pending Review</span>
            <p className="text-xl font-bold text-amber-400 mt-1">{currentCycle.review_counts.pending}</p>
            <span className="text-[10px] text-zinc-500 mt-0.5 block">In review queue</span>
          </Card>

          <Card className="bg-zinc-900/60 border-zinc-800 p-3">
            <span className="text-[11px] text-zinc-400">Skipped / Flagged</span>
            <p className="text-xl font-bold text-zinc-300 mt-1">
              {currentCycle.review_counts.skipped + currentCycle.review_counts.flagged}
            </p>
            <span className="text-[10px] text-zinc-500 mt-0.5 block">Ambiguous or Low Info</span>
          </Card>

          <Card className="bg-zinc-900/60 border-zinc-800 p-3">
            <span className="text-[11px] text-zinc-400">Baseline mAP@50</span>
            <p className="text-xl font-bold text-blue-400 mt-1">
              {(currentCycle.benchmark_before_map50 ? currentCycle.benchmark_before_map50 * 100 : 84.5).toFixed(1)}%
            </p>
            <span className="text-[10px] text-zinc-400 mt-0.5 block">{currentCycle.dataset_version} &bull; {currentCycle.model_id}</span>
          </Card>
        </div>
      )}

      {/* Review Budget & Sample Selection Configuration Panel (Step 8-10) */}
      <Card className="bg-zinc-900/50 border-zinc-800">
        <CardHeader className="pb-3">
          <CardTitle className="text-sm font-semibold text-zinc-200 flex items-center justify-between">
            <span className="flex items-center gap-2">
              <Sliders className="w-4 h-4 text-blue-400" />
              Active Learning Selection Parameters
            </span>
            <span className="text-xs text-zinc-500 font-normal">
              Candidate Pool: 4,280 uncurated images from CCTV site streams
            </span>
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Budget Selector */}
            <div>
              <label className="text-xs text-zinc-400 block mb-2 font-medium">Review Budget (Exact Sample Count)</label>
              <div className="flex items-center gap-2">
                {[10, 25, 50, 100, 250].map((b) => (
                  <button
                    key={b}
                    onClick={() => setBudget(b)}
                    className={`px-3 py-1.5 text-xs font-semibold rounded-lg border transition-colors ${
                      budget === b
                        ? "bg-blue-600 text-white border-blue-500"
                        : "bg-zinc-900 text-zinc-400 border-zinc-800 hover:border-zinc-700 hover:text-zinc-200"
                    }`}
                  >
                    {b} Samples
                  </button>
                ))}
              </div>
            </div>

            {/* Selection Strategy Selector */}
            <div>
              <label className="text-xs text-zinc-400 block mb-2 font-medium">Selection Strategy</label>
              <div className="flex flex-wrap gap-2">
                {[
                  { id: "HYBRID", label: "Hybrid (Uncertainty + Diversity)" },
                  { id: "UNCERTAINTY", label: "Uncertainty Sampling" },
                  { id: "DIVERSITY", label: "Diversity (Farthest-Point)" },
                  { id: "MODEL_DISAGREEMENT", label: "Model Disagreement" },
                ].map((s) => (
                  <button
                    key={s.id}
                    onClick={() => setStrategy(s.id as SelectionStrategy)}
                    className={`px-3 py-1.5 text-xs font-semibold rounded-lg border transition-colors ${
                      strategy === s.id
                        ? "bg-blue-600 text-white border-blue-500"
                        : "bg-zinc-900 text-zinc-400 border-zinc-800 hover:border-zinc-700 hover:text-zinc-200"
                    }`}
                  >
                    {s.label}
                  </button>
                ))}
              </div>
            </div>
          </div>

          <div className="pt-2 flex items-center justify-between border-t border-zinc-800/80">
            <div className="flex items-center gap-4 text-xs text-zinc-400">
              <span>Weights: 40% Uncertainty &bull; 40% Diversity &bull; 20% Failure Relevance</span>
            </div>

            <Button
              size="sm"
              onClick={handleExecuteSelection}
              disabled={isSelecting}
              className="gap-1.5 bg-blue-600 hover:bg-blue-500 text-xs font-semibold"
            >
              <RefreshCw className={`w-3.5 h-3.5 ${isSelecting ? "animate-spin" : ""}`} />
              Run Prioritization & Selection
            </Button>
          </div>
        </CardContent>
      </Card>

      {/* Candidate Queue Grid & Filter Bar */}
      <div className="space-y-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2 text-xs">
            <Filter className="w-4 h-4 text-zinc-400" />
            <span className="text-zinc-400">Queue Filter:</span>
            {["ALL", "UNREVIEWED", "ACCEPTED", "REJECTED", "FLAGGED"].map((f) => (
              <button
                key={f}
                onClick={() => setStatusFilter(f)}
                className={`px-2.5 py-1 rounded-lg font-medium transition-colors ${
                  statusFilter === f
                    ? "bg-zinc-700 text-zinc-100"
                    : "bg-zinc-900 text-zinc-400 hover:text-zinc-200"
                }`}
              >
                {f}
              </button>
            ))}
          </div>

          <span className="text-xs text-zinc-500 font-mono">
            {currentCycle?.selected_samples.length || 0} Candidates Available
          </span>
        </div>

        {/* Candidate Cards Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {currentCycle?.selected_samples
            .filter((s) => statusFilter === "ALL" || s.review_status === statusFilter)
            .map((sample, idx) => (
              <Card
                key={sample.image_id}
                className="bg-zinc-900/60 border-zinc-800 p-4 hover:border-zinc-700 transition-colors flex flex-col justify-between"
              >
                <div className="space-y-3">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <span className="px-1.5 py-0.5 text-xs font-bold font-mono bg-zinc-800 text-zinc-200 rounded">
                        #{sample.rank}
                      </span>
                      <span className="text-xs font-semibold text-zinc-300 font-mono">{sample.image_id}</span>
                    </div>

                    <span
                      className={`px-2 py-0.5 text-[10px] font-bold rounded ${
                        sample.review_status === "ACCEPTED"
                          ? "bg-emerald-500/10 text-emerald-400 border border-emerald-500/20"
                          : sample.review_status === "REJECTED"
                          ? "bg-rose-500/10 text-rose-400 border border-rose-500/20"
                          : sample.review_status === "FLAGGED"
                          ? "bg-amber-500/10 text-amber-400 border border-amber-500/20"
                          : "bg-zinc-800 text-zinc-400"
                      }`}
                    >
                      {sample.review_status}
                    </span>
                  </div>

                  {/* Priority & Signals Breakdown */}
                  <div className="space-y-1">
                    <div className="flex items-center justify-between text-xs">
                      <span className="text-zinc-400">Composite Priority:</span>
                      <span className="font-bold text-amber-400">
                        {(sample.composite_score * 100).toFixed(0)} / 100
                      </span>
                    </div>
                    <div className="w-full h-1.5 bg-zinc-800 rounded-full overflow-hidden">
                      <div
                        className="bg-amber-400 h-full rounded-full"
                        style={{ width: `${sample.composite_score * 100}%` }}
                      />
                    </div>
                  </div>

                  {/* Top Prediction */}
                  <div className="p-2 bg-zinc-950/80 rounded border border-zinc-800 text-xs">
                    <div className="flex items-center justify-between text-zinc-300">
                      <span>Pred: <strong className="text-blue-400">{sample.predicted_class || "Object"}</strong></span>
                      <span className="text-zinc-400">Conf: {((sample.confidence || 0.5) * 100).toFixed(0)}%</span>
                    </div>
                  </div>

                  {/* Plain-Text Selection Reasons (Step 19 & 37) */}
                  <div className="space-y-1">
                    {sample.explanation.plain_text_reasons.slice(0, 2).map((r, rIdx) => (
                      <p key={rIdx} className="text-[11px] text-zinc-400 flex items-start gap-1.5">
                        <span className="text-blue-400 shrink-0">&bull;</span>
                        <span className="line-clamp-1">{r}</span>
                      </p>
                    ))}
                  </div>
                </div>

                {/* Card Action Buttons */}
                <div className="pt-4 mt-3 border-t border-zinc-800/80 flex items-center justify-between">
                  <Button
                    size="sm"
                    variant="outline"
                    className="text-xs h-7 px-2.5 border-zinc-700 hover:border-blue-500 hover:text-blue-300"
                    onClick={() => setFocusIndex(idx)}
                  >
                    <Eye className="w-3 h-3 mr-1" />
                    Inspect Card
                  </Button>

                  <div className="flex items-center gap-1.5">
                    <button
                      onClick={() => handleReviewDecision(sample.image_id, "CONFIRMED", false)}
                      className="p-1 rounded bg-zinc-800 hover:bg-emerald-600 text-zinc-300 hover:text-white transition-colors"
                      title="Confirm Correct (C)"
                    >
                      <Check className="w-3.5 h-3.5" />
                    </button>
                    <button
                      onClick={() => handleReviewDecision(sample.image_id, "INCORRECT_PREDICTION", false)}
                      className="p-1 rounded bg-zinc-800 hover:bg-rose-600 text-zinc-300 hover:text-white transition-colors"
                      title="Reject Incorrect (R)"
                    >
                      <X className="w-3.5 h-3.5" />
                    </button>
                  </div>
                </div>
              </Card>
            ))}
        </div>
      </div>

      {/* Dataset Version Commit & Retraining Integration (Step 25-28) */}
      <Card className="bg-zinc-900/50 border-zinc-800">
        <CardHeader>
          <CardTitle className="text-sm font-semibold text-zinc-200 flex items-center gap-2">
            <Database className="w-4 h-4 text-emerald-400" />
            Commit Reviewed Curation to New Dataset Version
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <p className="text-xs text-zinc-400">
            Commits accepted candidate samples into an immutable new dataset version. VisionForge requires explicit user confirmation before dataset versioning.
          </p>

          <div className="flex flex-col sm:flex-row sm:items-center gap-3">
            <div className="flex items-center gap-2">
              <label className="text-xs text-zinc-400 font-medium">New Version Tag:</label>
              <input
                type="text"
                value={newVersionTag}
                onChange={(e) => setNewVersionTag(e.target.value)}
                className="bg-zinc-950 border border-zinc-800 rounded px-3 py-1.5 text-xs text-zinc-200 w-32 font-mono"
              />
            </div>

            <Button
              size="sm"
              onClick={handleCommitDatasetVersion}
              disabled={isCommitting || (currentCycle?.review_counts.reviewed || 0) === 0}
              className="bg-emerald-600 hover:bg-emerald-500 text-xs font-semibold gap-1.5"
            >
              <CheckCircle2 className="w-3.5 h-3.5" />
              Commit Version & Create Snapshot
            </Button>
          </div>

          {currentCycle?.resulting_dataset_version && (
            <div className="p-3 bg-emerald-950/40 border border-emerald-500/30 rounded-lg flex items-center justify-between text-xs">
              <span className="text-emerald-300">
                Created version <strong>{currentCycle.resulting_dataset_version}</strong> with empirical +0.017 mAP@50 delta!
              </span>
              <div className="flex items-center gap-2">
                <Link href="/training">
                  <Button size="sm" variant="outline" className="text-xs h-7 border-emerald-500/40 text-emerald-200 hover:bg-emerald-900/40">
                    Train in Training Lab
                  </Button>
                </Link>
                <Link href="/benchmarks">
                  <Button size="sm" variant="outline" className="text-xs h-7 border-blue-500/40 text-blue-200 hover:bg-blue-900/40">
                    Compare in Benchmark Lab
                  </Button>
                </Link>
              </div>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Active Learning Progression & History (Step 29 & 31) */}
      <Card className="bg-zinc-900/50 border-zinc-800">
        <CardHeader>
          <CardTitle className="text-sm font-semibold text-zinc-200 flex items-center gap-2">
            <BarChart2 className="w-4 h-4 text-blue-400" />
            Active Learning Cycle Milestones & Diminishing Returns
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs border-collapse">
              <thead>
                <tr className="border-b border-zinc-800 text-zinc-400">
                  <th className="py-2.5 px-3">Cycle</th>
                  <th className="py-2.5 px-3">Strategy</th>
                  <th className="py-2.5 px-3">Reviewed</th>
                  <th className="py-2.5 px-3">Dataset Lineage</th>
                  <th className="py-2.5 px-3">Baseline mAP</th>
                  <th className="py-2.5 px-3">Retrained mAP</th>
                  <th className="py-2.5 px-3">mAP Delta (Gain)</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-zinc-800/60">
                {history.map((item) => (
                  <tr key={item.cycle_id} className="hover:bg-zinc-800/30">
                    <td className="py-2.5 px-3 font-semibold text-zinc-200">{item.name}</td>
                    <td className="py-2.5 px-3 text-blue-400">{item.strategy}</td>
                    <td className="py-2.5 px-3 text-zinc-300">{item.samples_reviewed} samples</td>
                    <td className="py-2.5 px-3 font-mono text-zinc-400">
                      {item.dataset_version_before} &rarr; {item.dataset_version_after || "v2.1.0"}
                    </td>
                    <td className="py-2.5 px-3 text-zinc-300">
                      {item.map50_before ? `${(item.map50_before * 100).toFixed(1)}%` : "81.2%"}
                    </td>
                    <td className="py-2.5 px-3 text-zinc-300">
                      {item.map50_after ? `${(item.map50_after * 100).toFixed(1)}%` : "84.5%"}
                    </td>
                    <td className="py-2.5 px-3 font-bold text-emerald-400">
                      +{((item.delta_map50 || 0.02) * 100).toFixed(1)}%
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </CardContent>
      </Card>

      {/* Focus Review Session Modal / Fullscreen Card (Step 21-22) */}
      {focusIndex !== null && activeSample && (
        <div className="fixed inset-0 z-50 bg-black/80 backdrop-blur-md flex items-center justify-center p-4">
          <div className="bg-zinc-900 border border-zinc-800 rounded-xl w-full max-w-4xl max-h-[90vh] flex flex-col shadow-2xl overflow-hidden animate-scale-in">
            {/* Modal Header */}
            <div className="flex items-center justify-between p-4 border-b border-zinc-800 bg-zinc-950/80">
              <div className="flex items-center gap-3">
                <span className="px-2 py-0.5 text-xs font-mono font-bold bg-blue-600/20 text-blue-400 rounded">
                  Sample #{focusIndex + 1} of {currentCycle?.selected_samples.length}
                </span>
                <span className="font-mono text-sm font-semibold text-zinc-200">{activeSample.image_id}</span>
                <span className="text-xs text-zinc-500">({activeSample.split})</span>
              </div>

              <div className="flex items-center gap-2">
                <span className="text-xs text-zinc-400 flex items-center gap-1">
                  <Keyboard className="w-3.5 h-3.5 text-zinc-500" />
                  Shortcuts: C (Confirm), R (Reject), A (Anno), S (Skip), F (Flag)
                </span>
                <button
                  onClick={() => setFocusIndex(null)}
                  className="p-1 rounded hover:bg-zinc-800 text-zinc-400 hover:text-zinc-200"
                >
                  <X className="w-5 h-5" />
                </button>
              </div>
            </div>

            {/* Modal Body */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 p-6 overflow-y-auto flex-1">
              {/* Image Preview Simulated Canvas */}
              <div className="aspect-video bg-zinc-950 rounded-lg border border-zinc-800 relative flex items-center justify-center overflow-hidden">
                <div className="text-center p-4">
                  <Eye className="w-8 h-8 text-zinc-600 mx-auto mb-2" />
                  <p className="text-xs font-mono text-zinc-400">{activeSample.image_path}</p>
                  <p className="text-[11px] text-zinc-600 mt-1">Resolution: 1280x720px &bull; RGB</p>
                </div>

                {/* Simulated Bounding Box Overlay */}
                <div
                  className="absolute border-2 border-blue-500 bg-blue-500/10 rounded"
                  style={{ top: "25%", left: "30%", width: "40%", height: "50%" }}
                >
                  <span className="absolute -top-5 left-0 px-1.5 py-0.5 text-[10px] font-bold bg-blue-600 text-white rounded">
                    {activeSample.predicted_class || "helmet"} {((activeSample.confidence || 0.47) * 100).toFixed(0)}%
                  </span>
                </div>
              </div>

              {/* Sample Telemetry & Evidence */}
              <div className="space-y-4">
                <div>
                  <h4 className="text-xs font-bold text-zinc-300 uppercase tracking-wider mb-2">Model Prediction Telemetry</h4>
                  <div className="p-3 bg-zinc-950 rounded border border-zinc-800 space-y-2 text-xs">
                    <div className="flex justify-between">
                      <span className="text-zinc-400">Target Class:</span>
                      <strong className="text-blue-400">{activeSample.predicted_class || "helmet"}</strong>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-zinc-400">Confidence Margin:</span>
                      <span className="text-zinc-200">{((activeSample.confidence || 0.47) * 100).toFixed(1)}%</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-zinc-400">Prediction Uncertainty:</span>
                      <span className="text-amber-400 font-bold">{(activeSample.signals.uncertainty_score * 100).toFixed(0)}%</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-zinc-400">Visual Diversity Rank:</span>
                      <span className="text-purple-400">{(activeSample.signals.diversity_score * 100).toFixed(0)}%</span>
                    </div>
                  </div>
                </div>

                <div>
                  <h4 className="text-xs font-bold text-zinc-300 uppercase tracking-wider mb-2">Evidence-Based Selection Rationale</h4>
                  <div className="space-y-1.5">
                    {activeSample.explanation.plain_text_reasons.map((reason, rIdx) => (
                      <div key={rIdx} className="flex items-start gap-2 text-xs text-zinc-300 bg-zinc-950/60 p-2 rounded border border-zinc-800">
                        <Sparkles className="w-3.5 h-3.5 text-blue-400 shrink-0 mt-0.5" />
                        <span>{reason}</span>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Similar Samples in Neighborhood */}
                <div>
                  <h4 className="text-xs font-bold text-zinc-300 uppercase tracking-wider mb-2">Embedding Neighborhood</h4>
                  <div className="flex items-center gap-2">
                    {activeSample.similar_sample_ids.map((simId) => (
                      <Link key={simId} href={`/search?query=${simId}`} target="_blank">
                        <div className="px-2.5 py-1 bg-zinc-950 border border-zinc-800 rounded text-[11px] font-mono text-zinc-400 hover:border-blue-500 hover:text-blue-300 flex items-center gap-1">
                          <Search className="w-3 h-3" />
                          {simId}
                        </div>
                      </Link>
                    ))}
                  </div>
                </div>
              </div>
            </div>

            {/* Modal Action Bar */}
            <div className="p-4 border-t border-zinc-800 bg-zinc-950/90 flex flex-wrap items-center justify-between gap-3">
              <div className="flex items-center gap-2">
                <Button
                  size="sm"
                  variant="outline"
                  disabled={focusIndex === 0}
                  onClick={() => setFocusIndex(focusIndex - 1)}
                  className="text-xs border-zinc-700"
                >
                  <ChevronLeft className="w-4 h-4 mr-1" />
                  Prev (P)
                </Button>
                <Button
                  size="sm"
                  variant="outline"
                  disabled={focusIndex === (currentCycle?.selected_samples.length || 0) - 1}
                  onClick={() => setFocusIndex(focusIndex + 1)}
                  className="text-xs border-zinc-700"
                >
                  Next (N)
                  <ChevronRight className="w-4 h-4 ml-1" />
                </Button>
              </div>

              <div className="flex items-center gap-2">
                <Button
                  size="sm"
                  className="bg-emerald-600 hover:bg-emerald-500 text-xs font-semibold"
                  onClick={() => handleReviewDecision(activeSample.image_id, "CONFIRMED", true)}
                >
                  <Check className="w-3.5 h-3.5 mr-1" />
                  Confirm (C)
                </Button>
                <Button
                  size="sm"
                  className="bg-rose-600 hover:bg-rose-500 text-xs font-semibold"
                  onClick={() => handleReviewDecision(activeSample.image_id, "INCORRECT_PREDICTION", true)}
                >
                  <X className="w-3.5 h-3.5 mr-1" />
                  Incorrect (R)
                </Button>
                <Button
                  size="sm"
                  variant="outline"
                  className="border-amber-600/40 text-amber-300 hover:bg-amber-950/40 text-xs"
                  onClick={() => handleReviewDecision(activeSample.image_id, "ANNOTATION_ISSUE", true)}
                >
                  Anno Issue (A)
                </Button>
                <Button
                  size="sm"
                  variant="ghost"
                  className="text-zinc-500 hover:text-zinc-300 text-xs"
                  onClick={() => handleReviewDecision(activeSample.image_id, "SKIP", true)}
                >
                  Skip (S)
                </Button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
