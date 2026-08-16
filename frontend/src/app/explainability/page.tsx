"use client";

import React, { useEffect, useState } from "react";
import Link from "next/link";
import {
  Activity,
  AlertCircle,
  AlertTriangle,
  ArrowRight,
  BarChart2,
  Check,
  CheckCircle2,
  ChevronRight,
  Compass,
  Cpu,
  Database,
  Eye,
  FileText,
  Filter,
  Flame,
  GitCompare,
  HelpCircle,
  Info,
  Layers,
  Maximize2,
  MessageSquare,
  Network,
  Play,
  Plus,
  RefreshCw,
  Search,
  Send,
  ShieldAlert,
  Sliders,
  Sparkles,
  Tag,
  Target,
  ThumbsDown,
  ThumbsUp,
  Zap,
} from "lucide-react";
import { PageHeader } from "@/components/ui/PageHeader";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";

// ─── Interfaces ───────────────────────────────────────────────────

type ExplanationMethod = "GRAD_CAM" | "LAYER_CAM" | "INTEGRATED_GRADIENTS" | "ATTENTION_MAP" | "PERTURBATION";
type ExplanationStatus = "QUEUED" | "GENERATING" | "COMPLETED" | "FAILED" | "UNSUPPORTED";
type ReviewRating = "UNREVIEWED" | "USEFUL" | "NOT_USEFUL" | "UNCLEAR" | "NEEDS_INVESTIGATION";

interface AttributionArtifact {
  grid_width: number;
  grid_height: number;
  heatmap_grid: number[][];
  peak_intensity_coords: number[];
  mean_intensity: number;
  object_concentration_score: number;
  background_concentration_score: number;
  colormap: string;
}

interface StandardPrediction {
  prediction_id: string;
  class_id: number;
  class_name: string;
  confidence: number;
  bbox: {
    x_center: number;
    y_center: number;
    width: number;
    height: number;
  };
  model_id: string;
}

interface ExplanationRun {
  explanation_id: string;
  model_id: string;
  model_version: string;
  inference_id?: string;
  sample_id: string;
  image_path: string;
  dataset_id: string;
  dataset_version: string;
  split: string;
  method: ExplanationMethod;
  status: ExplanationStatus;
  target_class: string;
  prediction?: StandardPrediction;
  ground_truth_class?: string;
  is_correct_prediction?: boolean;
  artifact?: AttributionArtifact;
  diagnostic_summary: string;
  disclaimer: string;
  review_rating: ReviewRating;
  researcher_notes: string[];
  cache_hit: boolean;
  created_at: string;
  error_message?: string;
}

interface ExplanationComparison {
  comparison_id: string;
  explanation_a: ExplanationRun;
  explanation_b: ExplanationRun;
  attribution_difference_score: number;
  attribution_difference_grid: number[][];
  diagnostic_notes: string[];
}

export default function ModelExplainabilityWorkspacePage() {
  const [explanations, setExplanations] = useState<ExplanationRun[]>([]);
  const [selectedId, setSelectedId] = useState<string>("");
  const [currentExplanation, setCurrentExplanation] = useState<ExplanationRun | null>(null);
  const [viewMode, setViewMode] = useState<"original" | "attribution" | "overlay">("overlay");
  const [opacity, setOpacity] = useState<number>(55);
  const [colormap, setColormap] = useState<string>("jet");
  const [showBox, setShowBox] = useState<boolean>(true);
  const [selectedMethod, setSelectedMethod] = useState<ExplanationMethod>("GRAD_CAM");
  const [selectedTargetClass, setSelectedTargetClass] = useState<string>("helmet");

  // Researcher Notes & Review
  const [newNote, setNewNote] = useState<string>("");
  const [toastMessage, setToastMessage] = useState<string | null>(null);
  const [isGenerating, setIsGenerating] = useState<boolean>(false);

  // Side-by-side comparison
  const [comparison, setComparison] = useState<ExplanationComparison | null>(null);
  const [showComparison, setShowComparison] = useState<boolean>(false);

  useEffect(() => {
    loadExplanations();
  }, []);

  useEffect(() => {
    if (selectedId) {
      const found = explanations.find((e) => e.explanation_id === selectedId);
      if (found) {
        setCurrentExplanation(found);
        setSelectedMethod(found.method);
        setSelectedTargetClass(found.target_class);
      }
    }
  }, [selectedId, explanations]);

  const showToast = (msg: string) => {
    setToastMessage(msg);
    setTimeout(() => setToastMessage(null), 3000);
  };

  const loadExplanations = async () => {
    try {
      const res = await fetch("/api/v1/explainability/explanations");
      if (res.ok) {
        const json = await res.json();
        const list: ExplanationRun[] = json.data || [];
        setExplanations(list);
        if (list.length > 0 && !selectedId) {
          setSelectedId(list[0].explanation_id);
          setCurrentExplanation(list[0]);
          setSelectedMethod(list[0].method);
          setSelectedTargetClass(list[0].target_class);
        }
      }
    } catch (err) {
      console.error("Failed to load explanations:", err);
    }
  };

  const handleCreateExplanation = async () => {
    setIsGenerating(true);
    try {
      const res = await fetch("/api/v1/explainability/explanations", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model_id: currentExplanation?.model_id || "yolo11s.pt",
          sample_id: currentExplanation?.sample_id || "img_0007",
          target_class: selectedTargetClass,
          method: selectedMethod,
          config: {
            method: selectedMethod,
            colormap: colormap,
            opacity: opacity / 100.0,
            show_prediction_box: showBox,
          },
        }),
      });

      if (res.ok) {
        const json = await res.json();
        const run: ExplanationRun = json.data;
        showToast(
          run.cache_hit
            ? `Served cached attribution for ${run.target_class}`
            : `Generated ${run.method} attribution map!`
        );
        await loadExplanations();
        setSelectedId(run.explanation_id);
        setCurrentExplanation(run);
      }
    } catch (err) {
      console.error(err);
    } finally {
      setIsGenerating(false);
    }
  };

  const handleReviewRating = async (rating: ReviewRating) => {
    if (!currentExplanation) return;
    try {
      const res = await fetch(
        `/api/v1/explainability/explanations/${currentExplanation.explanation_id}/review`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ rating }),
        }
      );
      if (res.ok) {
        const json = await res.json();
        setCurrentExplanation(json.data);
        showToast(`Recorded rating: ${rating}`);
      }
    } catch (err) {
      console.error(err);
    }
  };

  const handleAddNote = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!currentExplanation || !newNote.trim()) return;
    try {
      const res = await fetch(
        `/api/v1/explainability/explanations/${currentExplanation.explanation_id}/notes`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ note: newNote }),
        }
      );
      if (res.ok) {
        const json = await res.json();
        setCurrentExplanation(json.data);
        setNewNote("");
        showToast("Researcher observation note saved!");
      }
    } catch (err) {
      console.error(err);
    }
  };

  const handleTriggerComparison = async () => {
    if (explanations.length < 2) return;
    const expA = explanations[0].explanation_id;
    const expB = explanations[1].explanation_id;
    try {
      const res = await fetch("/api/v1/explainability/compare", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          explanation_id_a: expA,
          explanation_id_b: expB,
        }),
      });
      if (res.ok) {
        const json = await res.json();
        setComparison(json.data);
        setShowComparison(true);
      }
    } catch (err) {
      console.error(err);
    }
  };

  return (
    <div className="space-y-6 pb-20">
      {/* Toast Notification */}
      {toastMessage && (
        <div className="fixed bottom-6 right-6 z-50 flex items-center gap-2 px-4 py-3 bg-emerald-950 border border-emerald-500 text-emerald-200 rounded-lg shadow-xl text-sm animate-fade-in">
          <CheckCircle2 className="w-4 h-4 text-emerald-400" />
          {toastMessage}
        </div>
      )}

      {/* Header & Selector */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <PageHeader
            title="Model Explainability & Visual Diagnostics Workspace"
            description="Spatial feature attribution, diagnostic evidence heatmaps, correct vs incorrect prediction divergence, and visual failure investigation."
          />
        </div>

        <div className="flex items-center gap-3">
          <div className="flex items-center bg-zinc-900 border border-zinc-800 rounded-lg px-3 py-1.5 gap-2">
            <Sparkles className="w-4 h-4 text-purple-400" />
            <select
              value={selectedId}
              onChange={(e) => setSelectedId(e.target.value)}
              className="bg-transparent text-sm font-medium text-zinc-200 focus:outline-none"
            >
              {explanations.map((exp) => (
                <option key={exp.explanation_id} value={exp.explanation_id}>
                  {exp.sample_id} ({exp.target_class} • {exp.method})
                </option>
              ))}
            </select>
          </div>

          <Button
            size="sm"
            onClick={handleTriggerComparison}
            variant="outline"
            className="text-xs h-8 border-purple-500/40 text-purple-300 gap-1.5"
          >
            <GitCompare className="w-3.5 h-3.5" />
            Compare Explanations
          </Button>
        </div>
      </div>

      {/* Explanation Context Bar */}
      {currentExplanation && (
        <div className="flex flex-wrap items-center justify-between gap-3 p-3 bg-zinc-900/60 border border-zinc-800 rounded-lg text-xs">
          <div className="flex flex-wrap items-center gap-3">
            <span className="flex items-center gap-1.5 text-zinc-400">
              <Cpu className="w-3.5 h-3.5 text-blue-400" />
              Model: <strong className="text-zinc-200">{currentExplanation.model_id} ({currentExplanation.model_version})</strong>
            </span>
            <span className="text-zinc-600">•</span>
            <span className="flex items-center gap-1.5 text-zinc-400">
              <Target className="w-3.5 h-3.5 text-emerald-400" />
              Sample: <strong className="text-zinc-200">{currentExplanation.sample_id}</strong>
            </span>
            <span className="text-zinc-600">•</span>
            <span className="flex items-center gap-1.5 text-zinc-400">
              <Tag className="w-3.5 h-3.5 text-purple-400" />
              Target Class: <strong className="text-purple-300">{currentExplanation.target_class}</strong>
            </span>
            <span className="text-zinc-600">•</span>
            <span className="flex items-center gap-1.5 text-zinc-400">
              Confidence: <strong className="text-blue-400">{((currentExplanation.prediction?.confidence || 0.8) * 100).toFixed(1)}%</strong>
            </span>
          </div>

          <div className="flex items-center gap-2">
            <span
              className={`px-2 py-0.5 rounded font-mono uppercase text-[10px] font-bold ${
                currentExplanation.is_correct_prediction
                  ? "bg-emerald-500/10 text-emerald-400 border border-emerald-500/20"
                  : "bg-rose-500/10 text-rose-400 border border-rose-500/20"
              }`}
            >
              {currentExplanation.is_correct_prediction ? "Correct Prediction" : "Incorrect (Failure)"}
            </span>
          </div>
        </div>
      )}

      {/* Main Workspace Layout */}
      {currentExplanation && (
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
          {/* Left Visualizer Canvas (7 Cols) */}
          <div className="lg:col-span-7 space-y-4">
            <Card className="bg-zinc-900/50 border-zinc-800 overflow-hidden">
              {/* Viewport Header & View Mode Switcher */}
              <div className="p-3 bg-zinc-950 border-b border-zinc-800 flex items-center justify-between">
                <div className="flex items-center gap-1 bg-zinc-900 p-1 rounded-lg border border-zinc-800 text-xs">
                  <button
                    onClick={() => setViewMode("original")}
                    className={`px-3 py-1 rounded font-medium transition-colors ${
                      viewMode === "original"
                        ? "bg-blue-600 text-white shadow-sm font-semibold"
                        : "text-zinc-400 hover:text-zinc-200"
                    }`}
                  >
                    Original Image
                  </button>
                  <button
                    onClick={() => setViewMode("attribution")}
                    className={`px-3 py-1 rounded font-medium transition-colors ${
                      viewMode === "attribution"
                        ? "bg-blue-600 text-white shadow-sm font-semibold"
                        : "text-zinc-400 hover:text-zinc-200"
                    }`}
                  >
                    Attribution Heatmap
                  </button>
                  <button
                    onClick={() => setViewMode("overlay")}
                    className={`px-3 py-1 rounded font-medium transition-colors ${
                      viewMode === "overlay"
                        ? "bg-blue-600 text-white shadow-sm font-semibold"
                        : "text-zinc-400 hover:text-zinc-200"
                    }`}
                  >
                    Overlay Blended
                  </button>
                </div>

                <div className="flex items-center gap-2 text-xs">
                  <span className="text-zinc-500 font-mono">Method:</span>
                  <span className="px-2 py-0.5 bg-purple-500/10 text-purple-300 rounded font-mono font-bold">
                    {currentExplanation.method}
                  </span>
                </div>
              </div>

              {/* Viewport Image & Heatmap Canvas */}
              <div className="aspect-video bg-zinc-950 relative flex items-center justify-center overflow-hidden select-none">
                {/* 1. Base Image Mock Canvas */}
                <div className="absolute inset-0 flex flex-col items-center justify-center p-6 bg-gradient-to-br from-zinc-900 via-zinc-950 to-black">
                  <Eye className="w-16 h-16 text-zinc-800 mb-2" />
                  <p className="text-xs font-mono text-zinc-600">Sample Stream Frame: {currentExplanation.sample_id}</p>
                  <p className="text-[10px] text-zinc-700 font-mono mt-0.5">Resolution: 1280 × 720 px</p>
                </div>

                {/* 2. Attribution Heatmap Visualization Layer */}
                {(viewMode === "attribution" || viewMode === "overlay") && currentExplanation.artifact && (
                  <div
                    className="absolute inset-0 pointer-events-none transition-opacity"
                    style={{
                      opacity: viewMode === "attribution" ? 1.0 : opacity / 100.0,
                      background:
                        colormap === "jet"
                          ? currentExplanation.is_correct_prediction
                            ? "radial-gradient(circle at 45% 40%, rgba(239, 68, 68, 0.85) 0%, rgba(234, 179, 8, 0.6) 35%, rgba(59, 130, 246, 0.3) 65%, transparent 85%)"
                            : "radial-gradient(circle at 75% 70%, rgba(239, 68, 68, 0.85) 0%, rgba(234, 179, 8, 0.6) 35%, rgba(59, 130, 246, 0.3) 65%, transparent 85%)"
                          : colormap === "viridis"
                          ? "radial-gradient(circle at 45% 40%, rgba(253, 231, 37, 0.85) 0%, rgba(33, 145, 140, 0.6) 40%, rgba(68, 1, 84, 0.3) 75%, transparent 90%)"
                          : "radial-gradient(circle at 45% 40%, rgba(252, 255, 164, 0.85) 0%, rgba(187, 55, 84, 0.6) 40%, rgba(0, 0, 4, 0.3) 75%, transparent 90%)",
                    }}
                  />
                )}

                {/* 3. Prediction Bounding Box Layer */}
                {showBox && (
                  <div
                    className="absolute border-2 border-emerald-400 bg-emerald-400/10 rounded pointer-events-none transition-all shadow-lg"
                    style={{
                      top: "22%",
                      left: "32%",
                      width: "30%",
                      height: "45%",
                    }}
                  >
                    <span className="absolute -top-5 left-0 px-1.5 py-0.5 text-[10px] font-bold bg-emerald-600 text-white rounded font-mono">
                      {currentExplanation.target_class} {((currentExplanation.prediction?.confidence || 0.8) * 100).toFixed(0)}%
                    </span>
                  </div>
                )}
              </div>

              {/* Viewport Interactive Controls Bar */}
              <div className="p-4 bg-zinc-950/80 border-t border-zinc-800 space-y-4">
                <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 text-xs">
                  {/* Opacity Control */}
                  <div>
                    <div className="flex justify-between text-zinc-400 mb-1 font-mono text-[11px]">
                      <span>Heatmap Opacity:</span>
                      <strong className="text-zinc-200">{opacity}%</strong>
                    </div>
                    <input
                      type="range"
                      min={0}
                      max={100}
                      value={opacity}
                      onChange={(e) => setOpacity(Number(e.target.value))}
                      className="w-full h-1.5 bg-zinc-800 rounded-lg appearance-none cursor-pointer accent-blue-500"
                    />
                  </div>

                  {/* Colormap Selector */}
                  <div>
                    <label className="text-zinc-400 block mb-1 font-mono text-[11px]">Colormap Palette:</label>
                    <select
                      value={colormap}
                      onChange={(e) => setColormap(e.target.value)}
                      className="w-full bg-zinc-900 border border-zinc-800 rounded px-2.5 py-1 text-xs text-zinc-200"
                    >
                      <option value="jet">Jet (Classic Scientific)</option>
                      <option value="viridis">Viridis (Perceptually Uniform)</option>
                      <option value="inferno">Inferno (High Contrast)</option>
                    </select>
                  </div>

                  {/* Prediction Box Toggle */}
                  <div className="flex items-end">
                    <label className="flex items-center gap-2 text-xs text-zinc-300 cursor-pointer bg-zinc-900 border border-zinc-800 px-3 py-1.5 rounded-lg w-full">
                      <input
                        type="checkbox"
                        checked={showBox}
                        onChange={(e) => setShowBox(e.target.checked)}
                        className="rounded border-zinc-700 text-blue-600 focus:ring-0"
                      />
                      <span>Show Prediction Box</span>
                    </label>
                  </div>
                </div>

                {/* Method & Target Recompute Bar */}
                <div className="pt-3 border-t border-zinc-800/80 flex flex-wrap items-center justify-between gap-3 text-xs">
                  <div className="flex items-center gap-2">
                    <select
                      value={selectedMethod}
                      onChange={(e) => setSelectedMethod(e.target.value as ExplanationMethod)}
                      className="bg-zinc-900 border border-zinc-800 rounded px-2.5 py-1 text-xs text-zinc-200"
                    >
                      <option value="GRAD_CAM">Grad-CAM (Gradient Weighted)</option>
                      <option value="LAYER_CAM">Layer-CAM (Fine-Grained)</option>
                      <option value="ATTENTION_MAP">Attention Map (ViT / SigLIP)</option>
                      <option value="INTEGRATED_GRADIENTS">Integrated Gradients</option>
                    </select>

                    <select
                      value={selectedTargetClass}
                      onChange={(e) => setSelectedTargetClass(e.target.value)}
                      className="bg-zinc-900 border border-zinc-800 rounded px-2.5 py-1 text-xs text-zinc-200"
                    >
                      <option value="helmet">Target: Helmet</option>
                      <option value="vest">Target: Vest</option>
                      <option value="person">Target: Person</option>
                      <option value="gloves">Target: Gloves</option>
                    </select>
                  </div>

                  <Button
                    size="sm"
                    onClick={handleCreateExplanation}
                    disabled={isGenerating}
                    className="bg-blue-600 hover:bg-blue-500 font-semibold text-xs gap-1.5 h-8"
                  >
                    <Play className="w-3.5 h-3.5" />
                    Recompute Attribution
                  </Button>
                </div>
              </div>
            </Card>
          </div>

          {/* Right Col: Diagnostics, Assessment, & Integrations (5 Cols) */}
          <div className="lg:col-span-5 space-y-4">
            {/* Diagnostic Evidence Summary */}
            <Card className="bg-zinc-900/50 border-zinc-800 p-4 space-y-3">
              <span className="text-xs font-bold text-zinc-300 uppercase tracking-wider flex items-center gap-2">
                <Info className="w-4 h-4 text-blue-400" />
                Diagnostic Evidence Summary
              </span>
              <p className="text-xs text-zinc-300 leading-relaxed bg-zinc-950 p-3 rounded-lg border border-zinc-800 font-mono">
                {currentExplanation.diagnostic_summary}
              </p>

              {/* Concentration Telemetry Scorecard */}
              {currentExplanation.artifact && (
                <div className="grid grid-cols-2 gap-3 pt-1">
                  <div className="p-3 bg-zinc-950 rounded-lg border border-zinc-800">
                    <span className="text-[10px] text-zinc-500 uppercase tracking-wider block font-mono">
                      Object Concentration
                    </span>
                    <p className="text-lg font-bold text-emerald-400 mt-0.5 font-mono">
                      {(currentExplanation.artifact.object_concentration_score * 100).toFixed(1)}%
                    </p>
                    <span className="text-[10px] text-zinc-400">Inside predicted box</span>
                  </div>

                  <div className="p-3 bg-zinc-950 rounded-lg border border-zinc-800">
                    <span className="text-[10px] text-zinc-500 uppercase tracking-wider block font-mono">
                      Background Energy
                    </span>
                    <p className="text-lg font-bold text-amber-400 mt-0.5 font-mono">
                      {(currentExplanation.artifact.background_concentration_score * 100).toFixed(1)}%
                    </p>
                    <span className="text-[10px] text-zinc-400">Outside target region</span>
                  </div>
                </div>
              )}

              {/* Scientific Validity Disclaimer */}
              <div className="p-3 rounded-lg bg-blue-950/30 border border-blue-500/30 text-[11px] text-blue-200/90 leading-relaxed flex items-start gap-2">
                <ShieldAlert className="w-4 h-4 text-blue-400 shrink-0 mt-0.5" />
                <span>{currentExplanation.disclaimer}</span>
              </div>
            </Card>

            {/* Ecosystem Integration Quick Actions */}
            <Card className="bg-zinc-900/50 border-zinc-800 p-4 space-y-3">
              <span className="text-xs font-bold text-zinc-300 uppercase tracking-wider block">
                Related Ecosystem Context
              </span>

              <div className="grid grid-cols-2 gap-2 text-xs">
                <Link href="/evaluation" className="w-full">
                  <Button variant="outline" size="sm" className="w-full justify-start gap-1.5 border-zinc-800 text-zinc-300 text-xs h-8">
                    <AlertCircle className="w-3.5 h-3.5 text-rose-400" />
                    View in Failure Gallery
                  </Button>
                </Link>

                <Link href="/search" className="w-full">
                  <Button variant="outline" size="sm" className="w-full justify-start gap-1.5 border-zinc-800 text-zinc-300 text-xs h-8">
                    <Search className="w-3.5 h-3.5 text-emerald-400" />
                    Find Similar Samples
                  </Button>
                </Link>

                <Link href="/explorer" className="w-full">
                  <Button variant="outline" size="sm" className="w-full justify-start gap-1.5 border-zinc-800 text-zinc-300 text-xs h-8">
                    <Compass className="w-3.5 h-3.5 text-purple-400" />
                    Embedding Context
                  </Button>
                </Link>

                <Link href="/datasets" className="w-full">
                  <Button variant="outline" size="sm" className="w-full justify-start gap-1.5 border-zinc-800 text-zinc-300 text-xs h-8">
                    <Database className="w-3.5 h-3.5 text-amber-400" />
                    Dataset Intelligence
                  </Button>
                </Link>
              </div>
            </Card>

            {/* Human Utility Assessment & Researcher Notes */}
            <Card className="bg-zinc-900/50 border-zinc-800 p-4 space-y-4">
              <div>
                <span className="text-xs font-bold text-zinc-300 uppercase tracking-wider block mb-2">
                  Human Assessment Review
                </span>
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-1.5">
                  {[
                    { key: "USEFUL", label: "Useful", color: "hover:bg-emerald-600 hover:text-white" },
                    { key: "NOT_USEFUL", label: "Not Useful", color: "hover:bg-rose-600 hover:text-white" },
                    { key: "UNCLEAR", label: "Unclear", color: "hover:bg-amber-600 hover:text-white" },
                    { key: "NEEDS_INVESTIGATION", label: "Investigate", color: "hover:bg-purple-600 hover:text-white" },
                  ].map((r) => (
                    <button
                      key={r.key}
                      onClick={() => handleReviewRating(r.key as ReviewRating)}
                      className={`px-2 py-1.5 rounded border text-[11px] font-semibold transition-colors ${
                        currentExplanation.review_rating === r.key
                          ? "bg-blue-600 border-blue-500 text-white"
                          : `bg-zinc-950 border-zinc-800 text-zinc-400 ${r.color}`
                      }`}
                    >
                      {r.label}
                    </button>
                  ))}
                </div>
              </div>

              {/* Researcher Notes Log */}
              <div>
                <span className="text-xs font-bold text-zinc-300 uppercase tracking-wider block mb-1.5">
                  Researcher Observations ({currentExplanation.researcher_notes.length})
                </span>
                {currentExplanation.researcher_notes.length > 0 && (
                  <div className="space-y-1.5 max-h-32 overflow-y-auto mb-2 pr-1">
                    {currentExplanation.researcher_notes.map((note, idx) => (
                      <div key={idx} className="p-2 bg-zinc-950 rounded border border-zinc-800 text-[11px] text-zinc-300 font-mono">
                        {note}
                      </div>
                    ))}
                  </div>
                )}

                <form onSubmit={handleAddNote} className="flex gap-2">
                  <input
                    type="text"
                    placeholder="Log researcher observation..."
                    value={newNote}
                    onChange={(e) => setNewNote(e.target.value)}
                    className="flex-1 bg-zinc-950 border border-zinc-800 rounded px-2.5 py-1 text-xs text-zinc-200"
                  />
                  <Button size="sm" type="submit" className="text-xs bg-zinc-800 hover:bg-zinc-700 text-zinc-200 h-7">
                    Add Note
                  </Button>
                </form>
              </div>
            </Card>
          </div>
        </div>
      )}

      {/* Side-by-Side Comparison Drawer */}
      {showComparison && comparison && (
        <Card className="bg-zinc-900/60 border-zinc-800 p-6 space-y-4 animate-scale-in">
          <div className="flex items-center justify-between border-b border-zinc-800 pb-3">
            <span className="text-sm font-bold text-zinc-200 flex items-center gap-2">
              <GitCompare className="w-4 h-4 text-purple-400" />
              Side-by-Side Explanation Comparison
            </span>
            <button onClick={() => setShowComparison(false)} className="text-zinc-500 hover:text-zinc-300">
              &times;
            </button>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="space-y-2 p-3 bg-zinc-950 rounded-lg border border-zinc-800">
              <strong className="text-xs text-blue-400 block font-mono">
                Sample A: {comparison.explanation_a.sample_id} ({comparison.explanation_a.target_class})
              </strong>
              <div className="aspect-video bg-zinc-900 rounded flex items-center justify-center">
                <Eye className="w-8 h-8 text-zinc-700" />
              </div>
              <p className="text-[11px] text-zinc-400 font-mono">{comparison.explanation_a.diagnostic_summary}</p>
            </div>

            <div className="space-y-2 p-3 bg-zinc-950 rounded-lg border border-zinc-800">
              <strong className="text-xs text-emerald-400 block font-mono">
                Sample B: {comparison.explanation_b.sample_id} ({comparison.explanation_b.target_class})
              </strong>
              <div className="aspect-video bg-zinc-900 rounded flex items-center justify-center">
                <Eye className="w-8 h-8 text-zinc-700" />
              </div>
              <p className="text-[11px] text-zinc-400 font-mono">{comparison.explanation_b.diagnostic_summary}</p>
            </div>
          </div>

          <div className="p-3 bg-zinc-950 rounded-lg border border-zinc-800 text-xs font-mono text-zinc-300">
            <strong>Comparative Notes:</strong> {comparison.diagnostic_notes.join(" ")}
          </div>
        </Card>
      )}

      {/* Explanation History Gallery Table */}
      <Card className="bg-zinc-900/50 border-zinc-800">
        <CardHeader>
          <CardTitle className="text-sm font-bold text-zinc-200 flex items-center gap-2">
            <Layers className="w-4 h-4 text-blue-400" />
            Generated Explanation Runs History ({explanations.length})
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="overflow-x-auto">
            <table className="w-full text-xs text-left">
              <thead>
                <tr className="border-b border-zinc-800 text-zinc-400 font-mono uppercase text-[10px]">
                  <th className="pb-2.5 font-bold">Sample ID</th>
                  <th className="pb-2.5 font-bold">Model</th>
                  <th className="pb-2.5 font-bold">Target Class</th>
                  <th className="pb-2.5 font-bold">Method</th>
                  <th className="pb-2.5 font-bold">Status</th>
                  <th className="pb-2.5 font-bold">Review Rating</th>
                  <th className="pb-2.5 font-bold">Cache Hit</th>
                  <th className="pb-2.5 font-bold text-right">Action</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-zinc-800/60 font-mono">
                {explanations.map((exp) => (
                  <tr
                    key={exp.explanation_id}
                    onClick={() => setSelectedId(exp.explanation_id)}
                    className={`cursor-pointer hover:bg-zinc-800/40 ${
                      selectedId === exp.explanation_id ? "bg-blue-600/10 font-bold" : ""
                    }`}
                  >
                    <td className="py-2.5 font-bold text-zinc-200">{exp.sample_id}</td>
                    <td className="py-2.5 text-zinc-400">{exp.model_id}</td>
                    <td className="py-2.5 text-purple-300 font-sans">{exp.target_class}</td>
                    <td className="py-2.5 text-blue-400">{exp.method}</td>
                    <td className="py-2.5">
                      <span className="px-2 py-0.5 rounded bg-emerald-500/10 text-emerald-400 text-[10px] font-bold">
                        {exp.status}
                      </span>
                    </td>
                    <td className="py-2.5 text-zinc-300">{exp.review_rating}</td>
                    <td className="py-2.5 text-zinc-500">{exp.cache_hit ? "Yes" : "No"}</td>
                    <td className="py-2.5 text-right text-blue-400 hover:underline">Inspect &rarr;</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
