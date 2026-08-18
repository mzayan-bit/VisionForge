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
  Eye,
  EyeOff,
  Filter,
  Layers,
  Maximize2,
  Play,
  RefreshCw,
  Search,
  Sliders,
  Sparkles,
  Upload,
  Zap,
} from "lucide-react";
import { PageHeader } from "@/components/ui/PageHeader";
import { Button } from "@/components/ui/Button";

// ─── Interfaces ───────────────────────────────────────────────────

interface BoundingBox {
  x_center: number;
  y_center: number;
  width: number;
  height: number;
  pixel_coords?: number[];
}

interface StandardPrediction {
  prediction_id: string;
  class_id: number;
  class_name: string;
  confidence: number;
  bbox: BoundingBox;
  model_id: string;
  model_version: string;
}

interface PredictionSummary {
  total_detections: number;
  classes_detected: string[];
  highest_confidence: number;
  average_confidence: number;
  inference_ms: number;
  model_id: string;
  image_width: number;
  image_height: number;
}

interface InferenceResult {
  inference_id: string;
  image_path: string;
  image_id?: string;
  model_id: string;
  model_version: string;
  predictions: StandardPrediction[];
  summary: PredictionSummary;
  config: {
    model_id: string;
    confidence_threshold: number;
    iou_threshold: number;
    imgsz: number;
    device: string;
  };
  visual_overlay_path?: string;
  created_at: string;
}

interface InferenceModelDescriptor {
  model_id: string;
  name: string;
  version: string;
  task: string;
  framework: string;
  checkpoint_path: string;
  status: string;
  training_run_id?: string;
  dataset_id?: string;
  map50?: number;
  precision?: number;
  recall?: number;
  is_available: boolean;
  unavailability_reason?: string;
}

interface ModelComparisonResult {
  comparison_id: string;
  image_path: string;
  image_width: number;
  image_height: number;
  model_a_result: InferenceResult;
  model_b_result: InferenceResult;
  notes: string;
  created_at: string;
}

interface InferenceBenchmarkResult {
  benchmark_id: string;
  model_id: string;
  model_version: string;
  device: string;
  runs: number;
  average_latency_ms: number;
  median_latency_ms: number;
  p95_latency_ms: number;
  min_latency_ms: number;
  max_latency_ms: number;
  fps: number;
  hardware_info: string;
  created_at: string;
}

// Color palette for category overlays
const CLASS_COLORS = [
  "#3B82F6", // blue
  "#10B981", // emerald
  "#F59E0B", // amber
  "#EF4444", // red
  "#8B5CF6", // purple
  "#EC4899", // pink
  "#06B6D4", // cyan
];

export default function VisionLabPage() {
  const router = useRouter();

  // Mode Selection State
  const [activeTab, setActiveTab] = useState<"inference" | "comparison" | "benchmark" | "history">(
    "inference"
  );

  // Model & Configuration State
  const [models, setModels] = useState<InferenceModelDescriptor[]>([]);
  const [selectedModelId, setSelectedModelId] = useState<string>("yolo11s.pt");
  const [confThreshold, setConfThreshold] = useState<number>(0.25);
  const [iouThreshold, setIouThreshold] = useState<number>(0.45);
  const [imgsz, setImgsz] = useState<number>(640);
  const [device, setDevice] = useState<string>("auto");

  // Input & Image State
  const [imagePath, setImagePath] = useState<string>("");
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [imageDimensions, setImageDimensions] = useState<{ width: number; height: number }>({
    width: 640,
    height: 480,
  });

  // Inference Execution State
  const [loading, setLoading] = useState<boolean>(false);
  const [loadingStage, setLoadingStage] = useState<string>("Ready");
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [inferenceResult, setInferenceResult] = useState<InferenceResult | null>(null);

  // Interactive Prediction Selection & Frontend Class Filtering
  const [selectedPrediction, setSelectedPrediction] = useState<StandardPrediction | null>(null);
  const [viewMode, setViewMode] = useState<"overlay" | "original" | "side_by_side">("overlay");
  const [hiddenClasses, setHiddenClasses] = useState<Set<string>>(new Set());
  const [minConfFilter, setMinConfFilter] = useState<number>(0.0);

  // Comparison State
  const [modelAId, setModelAId] = useState<string>("yolo11s.pt");
  const [modelBId, setModelBId] = useState<string>("rtdetr-l.pt");
  const [comparisonResult, setComparisonResult] = useState<ModelComparisonResult | null>(null);

  // Benchmark State
  const [bmRuns, setBmRuns] = useState<number>(20);
  const [bmWarmup, setBmWarmup] = useState<number>(3);
  const [benchmarkResult, setBenchmarkResult] = useState<InferenceBenchmarkResult | null>(null);

  // History State
  const [history, setHistory] = useState<InferenceResult[]>([]);

  // ─── Initial Load & Model Fetching ──────────────────────────────

  useEffect(() => {
    fetchModels();
    fetchHistory();
  }, []);

  const fetchModels = async () => {
    try {
      const res = await fetch("/api/v1/inference/models");
      if (res.ok) {
        const data = await res.json();
        setModels(data);
        if (data.length > 0 && !selectedModelId) {
          setSelectedModelId(data[0].model_id);
        }
      }
    } catch (err) {
      console.error("Failed to fetch inference models:", err);
    }
  };

  const fetchHistory = async () => {
    try {
      const res = await fetch("/api/v1/inference/history");
      if (res.ok) {
        const data = await res.json();
        setHistory(data);
      }
    } catch (err) {
      console.error("Failed to fetch inference history:", err);
    }
  };

  // ─── Image Upload Handler ──────────────────────────────────────

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    // Local preview URL
    const url = URL.createObjectURL(file);
    setPreviewUrl(url);

    const img = new Image();
    img.src = url;
    img.onload = () => {
      setImageDimensions({ width: img.width, height: img.height });
    };

    setLoading(true);
    setLoadingStage("Uploading target image...");
    setErrorMessage(null);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch("/api/v1/inference/upload", {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        const errorData = await res.json();
        throw new Error(errorData.detail || "Upload failed");
      }

      const data = await res.json();
      setImagePath(data.image_path);
    } catch (err: any) {
      setErrorMessage(err.message || "Failed to upload image.");
    } finally {
      setLoading(false);
      setLoadingStage("Ready");
    }
  };

  // Sample Image Loader
  const handleLoadSample = (sampleUrl: string) => {
    setPreviewUrl(sampleUrl);
    setImagePath(sampleUrl);
    setInferenceResult(null);
    setErrorMessage(null);
  };

  // ─── Run Single Image Inference ─────────────────────────────────

  const handleRunInference = async () => {
    if (!imagePath) {
      setErrorMessage("Please upload or select an image before running inference.");
      return;
    }

    setLoading(true);
    setLoadingStage("Warming model weights & preparing tensors...");
    setErrorMessage(null);

    const formData = new FormData();
    formData.append("image_path", imagePath);
    formData.append("model_id", selectedModelId);
    formData.append("confidence_threshold", confThreshold.toString());
    formData.append("iou_threshold", iouThreshold.toString());
    formData.append("imgsz", imgsz.toString());
    formData.append("device", device);

    try {
      setLoadingStage("Running model forward pass...");
      const res = await fetch("/api/v1/inference/run", {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        const errData = await res.json();
        throw new Error(errData.detail || "Inference execution failed.");
      }

      setLoadingStage("Processing bounding boxes & visual overlays...");
      const data: InferenceResult = await res.json();
      setInferenceResult(data);
      setSelectedPrediction(null);
      fetchHistory();
    } catch (err: any) {
      setErrorMessage(err.message || "Inference failed.");
    } finally {
      setLoading(false);
      setLoadingStage("Ready");
    }
  };

  // ─── Run Model Comparison ────────────────────────────────────────

  const handleRunComparison = async () => {
    if (!imagePath) {
      setErrorMessage("Please upload an image first for model comparison.");
      return;
    }

    setLoading(true);
    setLoadingStage("Running dual model inference comparison...");
    setErrorMessage(null);

    try {
      const res = await fetch("/api/v1/inference/compare", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          image_path: imagePath,
          model_a_id: modelAId,
          model_b_id: modelBId,
          config_a: {
            model_id: modelAId,
            confidence_threshold: confThreshold,
            iou_threshold: iouThreshold,
            imgsz,
            device,
          },
          config_b: {
            model_id: modelBId,
            confidence_threshold: confThreshold,
            iou_threshold: iouThreshold,
            imgsz,
            device,
          },
        }),
      });

      if (!res.ok) {
        const errData = await res.json();
        throw new Error(errData.detail || "Model comparison failed.");
      }

      const data: ModelComparisonResult = await res.json();
      setComparisonResult(data);
    } catch (err: any) {
      setErrorMessage(err.message || "Model comparison failed.");
    } finally {
      setLoading(false);
      setLoadingStage("Ready");
    }
  };

  // ─── Run Latency Benchmark ──────────────────────────────────────

  const handleRunBenchmark = async () => {
    setLoading(true);
    setLoadingStage(`Warming model & executing ${bmRuns} benchmark iterations...`);
    setErrorMessage(null);

    try {
      const res = await fetch("/api/v1/inference/benchmark", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model_id: selectedModelId,
          runs: bmRuns,
          warmup_runs: bmWarmup,
          batch_size: 1,
          imgsz,
          device,
        }),
      });

      if (!res.ok) {
        const errData = await res.json();
        throw new Error(errData.detail || "Benchmark execution failed.");
      }

      const data: InferenceBenchmarkResult = await res.json();
      setBenchmarkResult(data);
    } catch (err: any) {
      setErrorMessage(err.message || "Latency benchmark failed.");
    } finally {
      setLoading(false);
      setLoadingStage("Ready");
    }
  };

  // ─── Class Filtering Logic ───────────────────────────────────────

  const toggleClassVisibility = (className: string) => {
    setHiddenClasses((prev) => {
      const next = new Set(prev);
      if (next.has(className)) {
        next.delete(className);
      } else {
        next.add(className);
      }
      return next;
    });
  };

  const filteredPredictions = (inferenceResult?.predictions || []).filter((p) => {
    if (hiddenClasses.has(p.class_name)) return false;
    if (p.confidence < minConfFilter) return false;
    return true;
  });

  const selectedModelDescriptor = models.find((m) => m.model_id === selectedModelId);

  return (
    <div className="flex flex-col min-h-screen bg-[#0a0a0a] text-neutral-200 font-inter">
      {/* Workspace Header */}
      <PageHeader
        title="Vision Lab — Interactive Inference Studio"
        description="Flagship interactive workspace to run, inspect, compare, and benchmark trained Computer Vision models."
        breadcrumbs={["VisionForge", "Vision Lab"]}
        actions={
          <div className="flex items-center gap-2">
            <div className="flex items-center bg-[#141414] border border-white/10 rounded-lg p-1">
              <button
                onClick={() => setActiveTab("inference")}
                className={`px-3 py-1.5 rounded-md text-xs font-medium transition-all ${
                  activeTab === "inference"
                    ? "bg-blue-600/20 text-blue-400 border border-blue-500/30"
                    : "text-neutral-400 hover:text-white"
                }`}
              >
                Inference Studio
              </button>
              <button
                onClick={() => setActiveTab("comparison")}
                className={`px-3 py-1.5 rounded-md text-xs font-medium transition-all ${
                  activeTab === "comparison"
                    ? "bg-blue-600/20 text-blue-400 border border-blue-500/30"
                    : "text-neutral-400 hover:text-white"
                }`}
              >
                Compare Models
              </button>
              <button
                onClick={() => setActiveTab("benchmark")}
                className={`px-3 py-1.5 rounded-md text-xs font-medium transition-all ${
                  activeTab === "benchmark"
                    ? "bg-blue-600/20 text-blue-400 border border-blue-500/30"
                    : "text-neutral-400 hover:text-white"
                }`}
              >
                Latency Benchmark
              </button>
              <button
                onClick={() => setActiveTab("history")}
                className={`px-3 py-1.5 rounded-md text-xs font-medium transition-all ${
                  activeTab === "history"
                    ? "bg-blue-600/20 text-blue-400 border border-blue-500/30"
                    : "text-neutral-400 hover:text-white"
                }`}
              >
                History ({history.length})
              </button>
            </div>
          </div>
        }
      />

      {/* Main Container */}
      <div className="p-6 space-y-6 flex-1">
        {/* Error Alert Display */}
        {errorMessage && (
          <div className="bg-red-500/10 border border-red-500/30 text-red-400 px-4 py-3 rounded-lg flex items-center justify-between text-sm">
            <div className="flex items-center gap-2">
              <AlertCircle className="w-4 h-4 shrink-0" />
              <span>{errorMessage}</span>
            </div>
            <button
              onClick={() => setErrorMessage(null)}
              className="text-xs text-red-400/70 hover:text-red-300"
            >
              Dismiss
            </button>
          </div>
        )}

        {/* ─── TAB 1: SINGLE IMAGE INFERENCE STUDIO ───────────────────────── */}
        {activeTab === "inference" && (
          <div className="grid grid-cols-12 gap-6">
            {/* Control Sidebar Drawer (4 cols) */}
            <div className="col-span-12 lg:col-span-4 space-y-6">
              {/* 1. Model Selection Box */}
              <div className="bg-[#121212] border border-white/10 rounded-xl p-5 space-y-4">
                <div className="flex items-center justify-between">
                  <h3 className="text-xs font-semibold uppercase tracking-wider text-neutral-400 flex items-center gap-2">
                    <Cpu className="w-4 h-4 text-blue-400" />
                    Target Model
                  </h3>
                  <span className="text-[10px] px-2 py-0.5 rounded bg-blue-500/15 text-blue-400 border border-blue-500/30">
                    {selectedModelDescriptor?.framework || "Ultralytics"}
                  </span>
                </div>

                <div>
                  <label className="text-xs text-neutral-400 block mb-1 font-medium">
                    Select Model Checkpoint
                  </label>
                  <select
                    value={selectedModelId}
                    onChange={(e) => setSelectedModelId(e.target.value)}
                    className="w-full bg-[#1a1a1a] border border-white/10 rounded-lg px-3 py-2 text-xs font-medium text-white focus:outline-none focus:border-blue-500"
                  >
                    {models.map((m) => (
                      <option key={m.model_id} value={m.model_id} disabled={!m.is_available}>
                        {m.name} {!m.is_available ? "(Unavailable)" : ""}
                      </option>
                    ))}
                  </select>
                </div>

                {/* Model Metadata Card */}
                {selectedModelDescriptor && (
                  <div className="bg-[#181818] border border-white/5 rounded-lg p-3 text-xs space-y-1.5">
                    <div className="flex justify-between text-neutral-400">
                      <span>Task:</span>
                      <span className="text-white font-mono uppercase">
                        {selectedModelDescriptor.task}
                      </span>
                    </div>
                    <div className="flex justify-between text-neutral-400">
                      <span>Validation mAP@50:</span>
                      <span className="text-emerald-400 font-mono font-semibold">
                        {selectedModelDescriptor.map50 !== undefined &&
                        selectedModelDescriptor.map50 !== null
                          ? `${(selectedModelDescriptor.map50 * 100).toFixed(1)}%`
                          : "N/A"}
                      </span>
                    </div>
                    <div className="flex justify-between text-neutral-400">
                      <span>Status:</span>
                      <span className="text-blue-400 font-mono">
                        {selectedModelDescriptor.status}
                      </span>
                    </div>
                    {!selectedModelDescriptor.is_available && (
                      <div className="text-[11px] text-red-400 mt-1">
                        ⚠️ {selectedModelDescriptor.unavailability_reason}
                      </div>
                    )}
                  </div>
                )}
              </div>

              {/* 2. Image Input & Upload */}
              <div className="bg-[#121212] border border-white/10 rounded-xl p-5 space-y-4">
                <h3 className="text-xs font-semibold uppercase tracking-wider text-neutral-400 flex items-center gap-2">
                  <Upload className="w-4 h-4 text-emerald-400" />
                  Image Input
                </h3>

                <label className="relative block border-2 border-dashed border-white/10 hover:border-blue-500/50 rounded-xl p-6 text-center transition-colors cursor-pointer">
                  <Upload className="w-6 h-6 text-neutral-400 mx-auto mb-2" />
                  <p className="text-xs text-neutral-300 font-medium">
                    Drag & Drop or <span className="text-blue-400">Browse Image</span>
                  </p>
                  <p className="text-[10px] text-neutral-500 mt-1">Supports JPEG, PNG, WEBP</p>
                  <input
                    type="file"
                    accept="image/*"
                    onChange={handleFileUpload}
                    className="hidden"
                  />
                </label>

                {/* Quick Sample Selector */}
                <div>
                  <div className="text-[11px] text-neutral-400 mb-2 font-medium">
                    Or select sample test image:
                  </div>
                  <div className="flex gap-2">
                    <button
                      onClick={() =>
                        handleLoadSample(
                          "https://images.unsplash.com/photo-1578575437130-527eed3abbec?w=800"
                        )
                      }
                      className="px-2.5 py-1.5 bg-[#1a1a1a] hover:bg-[#252525] border border-white/10 rounded text-[11px] text-neutral-300 transition-colors"
                    >
                      Construction Site
                    </button>
                    <button
                      onClick={() =>
                        handleLoadSample(
                          "https://images.unsplash.com/photo-1504384308090-c894fdcc538d?w=800"
                        )
                      }
                      className="px-2.5 py-1.5 bg-[#1a1a1a] hover:bg-[#252525] border border-white/10 rounded text-[11px] text-neutral-300 transition-colors"
                    >
                      Industrial Floor
                    </button>
                  </div>
                </div>
              </div>

              {/* 3. Inference Hyperparameters */}
              <div className="bg-[#121212] border border-white/10 rounded-xl p-5 space-y-4">
                <h3 className="text-xs font-semibold uppercase tracking-wider text-neutral-400 flex items-center gap-2">
                  <Sliders className="w-4 h-4 text-purple-400" />
                  Inference Settings
                </h3>

                {/* Confidence Threshold Slider */}
                <div>
                  <div className="flex justify-between text-xs mb-1">
                    <span className="text-neutral-400">Confidence Threshold:</span>
                    <span className="font-mono text-blue-400 font-semibold">
                      {(confThreshold * 100).toFixed(0)}%
                    </span>
                  </div>
                  <input
                    type="range"
                    min={0.05}
                    max={0.95}
                    step={0.05}
                    value={confThreshold}
                    onChange={(e) => setConfThreshold(parseFloat(e.target.value))}
                    className="w-full accent-blue-500 bg-neutral-800"
                  />
                </div>

                {/* IoU NMS Threshold Slider */}
                <div>
                  <div className="flex justify-between text-xs mb-1">
                    <span className="text-neutral-400">IoU Threshold (NMS):</span>
                    <span className="font-mono text-purple-400 font-semibold">
                      {(iouThreshold * 100).toFixed(0)}%
                    </span>
                  </div>
                  <input
                    type="range"
                    min={0.1}
                    max={0.9}
                    step={0.05}
                    value={iouThreshold}
                    onChange={(e) => setIouThreshold(parseFloat(e.target.value))}
                    className="w-full accent-purple-500 bg-neutral-800"
                  />
                </div>

                {/* Image Resolution & Device */}
                <div className="grid grid-cols-2 gap-3">
                  <div>
                    <label className="text-[11px] text-neutral-400 block mb-1">Image Size</label>
                    <select
                      value={imgsz}
                      onChange={(e) => setImgsz(parseInt(e.target.value))}
                      className="w-full bg-[#1a1a1a] border border-white/10 rounded px-2.5 py-1.5 text-xs text-white"
                    >
                      <option value={320}>320 px</option>
                      <option value={640}>640 px (Default)</option>
                      <option value={1024}>1024 px</option>
                    </select>
                  </div>
                  <div>
                    <label className="text-[11px] text-neutral-400 block mb-1">Compute Target</label>
                    <select
                      value={device}
                      onChange={(e) => setDevice(e.target.value)}
                      className="w-full bg-[#1a1a1a] border border-white/10 rounded px-2.5 py-1.5 text-xs text-white"
                    >
                      <option value="auto">Auto Select</option>
                      <option value="cpu">CPU Only</option>
                      <option value="cuda">NVIDIA CUDA</option>
                      <option value="mps">Apple MPS (M4)</option>
                    </select>
                  </div>
                </div>

                {/* Execute Button */}
                <Button
                  onClick={handleRunInference}
                  disabled={loading || !imagePath || !selectedModelDescriptor?.is_available}
                  className="w-full bg-blue-600 hover:bg-blue-500 text-white font-medium py-2.5 rounded-lg flex items-center justify-center gap-2 transition-all shadow-lg shadow-blue-600/20"
                >
                  {loading ? (
                    <>
                      <RefreshCw className="w-4 h-4 animate-spin" />
                      <span>{loadingStage}</span>
                    </>
                  ) : (
                    <>
                      <Play className="w-4 h-4 fill-current" />
                      <span>Run Inference</span>
                    </>
                  )}
                </Button>
              </div>
            </div>

            {/* Main Interactive Viewer & Telemetry (8 cols) */}
            <div className="col-span-12 lg:col-span-8 space-y-6">
              {/* Telemetry Summary Bar */}
              {inferenceResult && (
                <div className="grid grid-cols-4 gap-4">
                  <div className="bg-[#121212] border border-white/10 rounded-xl p-4">
                    <div className="text-[11px] font-medium text-neutral-400 uppercase tracking-wider mb-1">
                      Detections
                    </div>
                    <div className="text-2xl font-bold font-mono text-white">
                      {filteredPredictions.length}
                    </div>
                    <div className="text-[10px] text-neutral-500 mt-1">
                      Across {new Set(filteredPredictions.map((p) => p.class_name)).size} categories
                    </div>
                  </div>

                  <div className="bg-[#121212] border border-white/10 rounded-xl p-4">
                    <div className="text-[11px] font-medium text-neutral-400 uppercase tracking-wider mb-1">
                      Max Confidence
                    </div>
                    <div className="text-2xl font-bold font-mono text-emerald-400">
                      {(
                        maxConfidence(filteredPredictions) * 100
                      ).toFixed(1)}
                      %
                    </div>
                    <div className="text-[10px] text-neutral-500 mt-1">Highest detection score</div>
                  </div>

                  <div className="bg-[#121212] border border-white/10 rounded-xl p-4">
                    <div className="text-[11px] font-medium text-neutral-400 uppercase tracking-wider mb-1">
                      Inference Latency
                    </div>
                    <div className="text-2xl font-bold font-mono text-blue-400">
                      {inferenceResult.summary.inference_ms} <span className="text-xs font-normal">ms</span>
                    </div>
                    <div className="text-[10px] text-neutral-500 mt-1">
                      {(1000 / inferenceResult.summary.inference_ms).toFixed(1)} FPS throughput
                    </div>
                  </div>

                  <div className="bg-[#121212] border border-white/10 rounded-xl p-4">
                    <div className="text-[11px] font-medium text-neutral-400 uppercase tracking-wider mb-1">
                      Resolution
                    </div>
                    <div className="text-lg font-bold font-mono text-white">
                      {inferenceResult.summary.image_width}x{inferenceResult.summary.image_height}
                    </div>
                    <div className="text-[10px] text-neutral-500 mt-1 font-mono">
                      {inferenceResult.config.model_id}
                    </div>
                  </div>
                </div>
              )}

              {/* Interactive Prediction Canvas */}
              <div className="bg-[#121212] border border-white/10 rounded-xl p-5 space-y-4">
                <div className="flex items-center justify-between border-b border-white/10 pb-3">
                  <div className="flex items-center gap-3">
                    <h3 className="text-sm font-semibold text-white">Visual Prediction Canvas</h3>
                    <span className="text-xs text-neutral-400 font-mono">
                      {filteredPredictions.length} visible detections
                    </span>
                  </div>

                  {/* View Controls */}
                  <div className="flex items-center bg-[#1a1a1a] border border-white/10 rounded-lg p-1">
                    <button
                      onClick={() => setViewMode("overlay")}
                      className={`px-2.5 py-1 text-xs rounded font-medium transition-colors ${
                        viewMode === "overlay"
                          ? "bg-blue-600/30 text-blue-400 border border-blue-500/30"
                          : "text-neutral-400 hover:text-white"
                      }`}
                    >
                      Interactive Overlay
                    </button>
                    <button
                      onClick={() => setViewMode("original")}
                      className={`px-2.5 py-1 text-xs rounded font-medium transition-colors ${
                        viewMode === "original"
                          ? "bg-blue-600/30 text-blue-400 border border-blue-500/30"
                          : "text-neutral-400 hover:text-white"
                      }`}
                    >
                      Original
                    </button>
                  </div>
                </div>

                {/* Display Canvas Box */}
                <div className="relative min-h-[420px] bg-[#080808] border border-white/5 rounded-lg overflow-hidden flex items-center justify-center">
                  {previewUrl ? (
                    <div className="relative inline-block max-w-full max-h-[600px]">
                      {/* Base Image */}
                      <img
                        src={previewUrl}
                        alt="Inference Target"
                        className="max-w-full max-h-[550px] object-contain rounded"
                      />

                      {/* SVG Interactive Bounding Boxes Overlay */}
                      {viewMode === "overlay" && inferenceResult && (
                        <svg
                          className="absolute inset-0 w-full h-full pointer-events-none"
                          viewBox={`0 0 ${imageDimensions.width || 640} ${
                            imageDimensions.height || 480
                          }`}
                        >
                          {filteredPredictions.map((pred) => {
                            const color = CLASS_COLORS[pred.class_id % CLASS_COLORS.length];
                            const isSelected =
                              selectedPrediction?.prediction_id === pred.prediction_id;

                            let x1 = 0,
                              y1 = 0,
                              bw = 0,
                              bh = 0;

                            if (pred.bbox.pixel_coords && pred.bbox.pixel_coords.length === 4) {
                              x1 = pred.bbox.pixel_coords[0];
                              y1 = pred.bbox.pixel_coords[1];
                              bw = pred.bbox.pixel_coords[2] - x1;
                              bh = pred.bbox.pixel_coords[3] - y1;
                            } else {
                              x1 =
                                (pred.bbox.x_center - pred.bbox.width / 2) *
                                (imageDimensions.width || 640);
                              y1 =
                                (pred.bbox.y_center - pred.bbox.height / 2) *
                                (imageDimensions.height || 480);
                              bw = pred.bbox.width * (imageDimensions.width || 640);
                              bh = pred.bbox.height * (imageDimensions.height || 480);
                            }

                            return (
                              <g
                                key={pred.prediction_id}
                                className="pointer-events-auto cursor-pointer"
                                onClick={() => setSelectedPrediction(pred)}
                              >
                                <rect
                                  x={x1}
                                  y={y1}
                                  width={bw}
                                  height={bh}
                                  fill={isSelected ? `${color}33` : "transparent"}
                                  stroke={color}
                                  strokeWidth={isSelected ? 4 : 2.5}
                                  className="transition-all hover:opacity-80"
                                />
                                <rect
                                  x={x1}
                                  y={Math.max(0, y1 - 22)}
                                  width={Math.max(80, pred.class_name.length * 9 + 40)}
                                  height={22}
                                  fill={color}
                                  rx={3}
                                />
                                <text
                                  x={x1 + 6}
                                  y={Math.max(14, y1 - 6)}
                                  fill="#ffffff"
                                  fontSize="12"
                                  fontWeight="600"
                                  fontFamily="monospace"
                                >
                                  {pred.class_name} {(pred.confidence * 100).toFixed(0)}%
                                </text>
                              </g>
                            );
                          })}
                        </svg>
                      )}
                    </div>
                  ) : (
                    <div className="text-center py-16 text-neutral-500 space-y-2">
                      <Sparkles className="w-8 h-8 text-neutral-600 mx-auto" />
                      <div className="text-xs font-medium">No Image Uploaded</div>
                      <div className="text-[11px] text-neutral-600">
                        Upload or select an image to run real vision model inference.
                      </div>
                    </div>
                  )}
                </div>

                {/* Integration Actions Toolbar */}
                {inferenceResult && (
                  <div className="flex flex-wrap items-center justify-between gap-3 border-t border-white/10 pt-4">
                    <div className="flex items-center gap-2">
                      <Link href={`/search?query_id=${inferenceResult.inference_id}`}>
                        <Button
                          variant="secondary"
                          size="sm"
                          icon={<Search className="w-3.5 h-3.5 text-emerald-400" />}
                        >
                          Find Similar Images
                        </Button>
                      </Link>

                      <Link href="/explorer">
                        <Button
                          variant="secondary"
                          size="sm"
                          icon={<Layers className="w-3.5 h-3.5 text-purple-400" />}
                        >
                          Open Embedding Explorer
                        </Button>
                      </Link>
                    </div>

                    <div className="flex items-center gap-2 text-xs text-neutral-400 font-mono">
                      <span>ID: {inferenceResult.inference_id}</span>
                    </div>
                  </div>
                )}
              </div>

              {/* Detections List & Frontend Class Filter */}
              {inferenceResult && (
                <div className="bg-[#121212] border border-white/10 rounded-xl p-5 space-y-4">
                  <div className="flex items-center justify-between">
                    <h3 className="text-xs font-semibold uppercase tracking-wider text-neutral-400 flex items-center gap-2">
                      <Filter className="w-4 h-4 text-blue-400" />
                      Frontend Category Filters & Inspector
                    </h3>
                    <div className="text-xs text-neutral-500">
                      Toggle categories to filter visually without re-running model
                    </div>
                  </div>

                  {/* Category Pills */}
                  <div className="flex flex-wrap gap-2">
                    {inferenceResult.summary.classes_detected.map((cls, idx) => {
                      const color = CLASS_COLORS[idx % CLASS_COLORS.length];
                      const isHidden = hiddenClasses.has(cls);
                      return (
                        <button
                          key={cls}
                          onClick={() => toggleClassVisibility(cls)}
                          className={`px-3 py-1.5 rounded-lg text-xs font-medium border flex items-center gap-2 transition-all cursor-pointer ${
                            isHidden
                              ? "bg-neutral-900 border-neutral-800 text-neutral-600 line-through"
                              : "bg-[#181818] border-white/10 text-white hover:border-white/20"
                          }`}
                        >
                          <span
                            className="w-2.5 h-2.5 rounded-full"
                            style={{ backgroundColor: color }}
                          />
                          <span>{cls}</span>
                          {isHidden ? (
                            <EyeOff className="w-3 h-3 text-neutral-600" />
                          ) : (
                            <Eye className="w-3 h-3 text-neutral-400" />
                          )}
                        </button>
                      );
                    })}
                  </div>

                  {/* Detailed Inspector Drawer for Clicked Box */}
                  {selectedPrediction && (
                    <div className="bg-[#181818] border border-blue-500/30 rounded-xl p-4 text-xs space-y-2 relative">
                      <button
                        onClick={() => setSelectedPrediction(null)}
                        className="absolute top-3 right-3 text-neutral-500 hover:text-white"
                      >
                        ✕
                      </button>
                      <div className="font-semibold text-blue-400 flex items-center gap-2">
                        <CheckCircle2 className="w-4 h-4" />
                        Selected Detection Details
                      </div>
                      <div className="grid grid-cols-3 gap-2 font-mono text-neutral-300 pt-1">
                        <div>Class: <span className="text-white font-bold">{selectedPrediction.class_name}</span></div>
                        <div>Confidence: <span className="text-emerald-400">{(selectedPrediction.confidence * 100).toFixed(1)}%</span></div>
                        <div>Model: <span className="text-white">{selectedPrediction.model_id}</span></div>
                        <div>BBox Center: [{selectedPrediction.bbox.x_center}, {selectedPrediction.bbox.y_center}]</div>
                        <div>BBox Dimensions: [{selectedPrediction.bbox.width} x {selectedPrediction.bbox.height}]</div>
                        <div>Prediction ID: {selectedPrediction.prediction_id}</div>
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>
        )}

        {/* ─── TAB 2: MODEL COMPARISON MODE ───────────────────────────────── */}
        {activeTab === "comparison" && (
          <div className="space-y-6">
            <div className="bg-[#121212] border border-white/10 rounded-xl p-5 space-y-4">
              <h3 className="text-xs font-semibold uppercase tracking-wider text-neutral-400 flex items-center gap-2">
                <BarChart2 className="w-4 h-4 text-blue-400" />
                Side-by-Side Model Comparison Setup
              </h3>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <label className="text-xs text-neutral-400 block mb-1">Model A (Baseline)</label>
                  <select
                    value={modelAId}
                    onChange={(e) => setModelAId(e.target.value)}
                    className="w-full bg-[#1a1a1a] border border-white/10 rounded px-3 py-2 text-xs text-white"
                  >
                    {models.map((m) => (
                      <option key={`a_${m.model_id}`} value={m.model_id}>
                        {m.name}
                      </option>
                    ))}
                  </select>
                </div>
                <div>
                  <label className="text-xs text-neutral-400 block mb-1">Model B (Candidate)</label>
                  <select
                    value={modelBId}
                    onChange={(e) => setModelBId(e.target.value)}
                    className="w-full bg-[#1a1a1a] border border-white/10 rounded px-3 py-2 text-xs text-white"
                  >
                    {models.map((m) => (
                      <option key={`b_${m.model_id}`} value={m.model_id}>
                        {m.name}
                      </option>
                    ))}
                  </select>
                </div>
              </div>

              <Button
                onClick={handleRunComparison}
                disabled={loading || !imagePath}
                className="bg-blue-600 hover:bg-blue-500 text-white font-medium px-5 py-2 rounded-lg flex items-center gap-2"
              >
                {loading ? <RefreshCw className="w-4 h-4 animate-spin" /> : <Play className="w-4 h-4" />}
                <span>Execute Model Comparison</span>
              </Button>
            </div>

            {/* Comparison Side-by-Side Results */}
            {comparisonResult && (
              <div className="space-y-6">
                {/* Notes Summary Card */}
                <div className="bg-[#121212] border border-blue-500/30 rounded-xl p-4 text-xs font-mono text-neutral-300">
                  <span className="text-blue-400 font-bold uppercase mr-2">Analysis:</span>
                  {comparisonResult.notes}
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  {/* Model A Results */}
                  <div className="bg-[#121212] border border-white/10 rounded-xl p-5 space-y-3">
                    <div className="flex justify-between items-center border-b border-white/10 pb-2">
                      <h4 className="font-semibold text-sm text-white">Model A: {comparisonResult.model_a_result.model_id}</h4>
                      <span className="text-xs font-mono text-blue-400">{comparisonResult.model_a_result.summary.inference_ms} ms</span>
                    </div>
                    <div className="text-xs text-neutral-400">
                      Detections: <span className="text-white font-bold">{comparisonResult.model_a_result.summary.total_detections}</span> | Max Conf: <span className="text-emerald-400">{(comparisonResult.model_a_result.summary.highest_confidence * 100).toFixed(1)}%</span>
                    </div>
                  </div>

                  {/* Model B Results */}
                  <div className="bg-[#121212] border border-white/10 rounded-xl p-5 space-y-3">
                    <div className="flex justify-between items-center border-b border-white/10 pb-2">
                      <h4 className="font-semibold text-sm text-white">Model B: {comparisonResult.model_b_result.model_id}</h4>
                      <span className="text-xs font-mono text-purple-400">{comparisonResult.model_b_result.summary.inference_ms} ms</span>
                    </div>
                    <div className="text-xs text-neutral-400">
                      Detections: <span className="text-white font-bold">{comparisonResult.model_b_result.summary.total_detections}</span> | Max Conf: <span className="text-emerald-400">{(comparisonResult.model_b_result.summary.highest_confidence * 100).toFixed(1)}%</span>
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}

        {/* ─── TAB 3: LATENCY BENCHMARK MODE ──────────────────────────────── */}
        {activeTab === "benchmark" && (
          <div className="space-y-6">
            <div className="bg-[#121212] border border-white/10 rounded-xl p-5 space-y-4 max-w-2xl">
              <h3 className="text-xs font-semibold uppercase tracking-wider text-neutral-400 flex items-center gap-2">
                <Activity className="w-4 h-4 text-emerald-400" />
                Inference Latency & Throughput Benchmark
              </h3>

              <div className="grid grid-cols-2 gap-4 text-xs">
                <div>
                  <label className="text-neutral-400 block mb-1">Target Model</label>
                  <select
                    value={selectedModelId}
                    onChange={(e) => setSelectedModelId(e.target.value)}
                    className="w-full bg-[#1a1a1a] border border-white/10 rounded px-3 py-2 text-white"
                  >
                    {models.map((m) => (
                      <option key={`bm_${m.model_id}`} value={m.model_id}>
                        {m.name}
                      </option>
                    ))}
                  </select>
                </div>

                <div>
                  <label className="text-neutral-400 block mb-1">Benchmark Iterations</label>
                  <input
                    type="number"
                    value={bmRuns}
                    onChange={(e) => setBmRuns(parseInt(e.target.value))}
                    className="w-full bg-[#1a1a1a] border border-white/10 rounded px-3 py-2 text-white font-mono"
                  />
                </div>
              </div>

              <Button
                onClick={handleRunBenchmark}
                disabled={loading}
                className="bg-emerald-600 hover:bg-emerald-500 text-white font-medium px-5 py-2 rounded-lg flex items-center gap-2"
              >
                {loading ? <RefreshCw className="w-4 h-4 animate-spin" /> : <Zap className="w-4 h-4" />}
                <span>Execute Benchmark</span>
              </Button>
            </div>

            {/* Benchmark Output Card */}
            {benchmarkResult && (
              <div className="bg-[#121212] border border-emerald-500/30 rounded-xl p-6 space-y-6">
                <h4 className="text-sm font-semibold text-white">Benchmark Telemetry Results</h4>
                <div className="grid grid-cols-4 gap-4 text-center">
                  <div className="bg-[#181818] p-4 rounded-lg border border-white/5">
                    <div className="text-xs text-neutral-400">Avg Latency</div>
                    <div className="text-2xl font-bold font-mono text-emerald-400 mt-1">
                      {benchmarkResult.average_latency_ms} ms
                    </div>
                  </div>
                  <div className="bg-[#181818] p-4 rounded-lg border border-white/5">
                    <div className="text-xs text-neutral-400">Median Latency (p50)</div>
                    <div className="text-2xl font-bold font-mono text-blue-400 mt-1">
                      {benchmarkResult.median_latency_ms} ms
                    </div>
                  </div>
                  <div className="bg-[#181818] p-4 rounded-lg border border-white/5">
                    <div className="text-xs text-neutral-400">p95 Latency</div>
                    <div className="text-2xl font-bold font-mono text-purple-400 mt-1">
                      {benchmarkResult.p95_latency_ms} ms
                    </div>
                  </div>
                  <div className="bg-[#181818] p-4 rounded-lg border border-white/5">
                    <div className="text-xs text-neutral-400">Throughput</div>
                    <div className="text-2xl font-bold font-mono text-amber-400 mt-1">
                      {benchmarkResult.fps} FPS
                    </div>
                  </div>
                </div>

                <div className="text-xs font-mono text-neutral-400 border-t border-white/10 pt-3">
                  Environment: {benchmarkResult.hardware_info}
                </div>
              </div>
            )}
          </div>
        )}

        {/* ─── TAB 4: INFERENCE HISTORY ──────────────────────────────────── */}
        {activeTab === "history" && (
          <div className="bg-[#121212] border border-white/10 rounded-xl overflow-hidden">
            <div className="p-4 border-b border-white/10 flex justify-between items-center">
              <h3 className="text-xs font-semibold uppercase tracking-wider text-neutral-400">
                Inference Execution History
              </h3>
              <Button variant="secondary" size="sm" onClick={fetchHistory}>
                Refresh History
              </Button>
            </div>

            <div className="divide-y divide-white/5">
              {history.map((rec) => (
                <div
                  key={rec.inference_id}
                  className="p-4 hover:bg-[#181818] transition-colors flex items-center justify-between text-xs"
                >
                  <div className="space-y-1">
                    <div className="font-mono text-blue-400 font-semibold">{rec.inference_id}</div>
                    <div className="text-neutral-400">
                      Model: <span className="text-white">{rec.model_id}</span> | Image:{" "}
                      <span className="text-neutral-300 font-mono">
                        {rec.image_path.split("/").pop()}
                      </span>
                    </div>
                  </div>

                  <div className="flex items-center gap-6">
                    <div className="text-right">
                      <div className="font-mono text-emerald-400 font-semibold">
                        {rec.summary.total_detections} Detections
                      </div>
                      <div className="text-neutral-500">{rec.summary.inference_ms} ms</div>
                    </div>

                    <button
                      onClick={() => {
                        setInferenceResult(rec);
                        setPreviewUrl(rec.image_path);
                        setActiveTab("inference");
                      }}
                      className="px-3 py-1.5 bg-blue-600/20 text-blue-400 border border-blue-500/30 rounded-lg hover:bg-blue-600/30 transition-colors"
                    >
                      Re-inspect
                    </button>
                  </div>
                </div>
              ))}

              {history.length === 0 && (
                <div className="p-8 text-center text-xs text-neutral-500">
                  No inference runs recorded yet. Run inference on an image above to populate history.
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

// ─── Helper Functions ────────────────────────────────────────────────

function maxConfidence(preds: StandardPrediction[]): number {
  if (!preds || preds.length === 0) return 0.0;
  return Math.max(...preds.map((p) => p.confidence));
}
