"use client";

import React, { useEffect, useState } from "react";
import {
  Upload,
  Sparkles,
  Cpu,
  Zap,
  Activity,
  HardDrive,
  CheckCircle2,
  AlertCircle,
  RefreshCw,
  Trash2,
  FileImage,
  BarChart3,
  Layers,
  Hash,
} from "lucide-react";
import { PageHeader } from "@/components/ui/PageHeader";
import { Card } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";
import { Badge } from "@/components/ui/Badge";

interface ImageMetadata {
  width: number;
  height: number;
  format: string;
  mode: string;
  aspect_ratio: number;
  file_size_bytes: number;
}

interface VectorStats {
  min: number;
  max: number;
  mean: number;
  std: number;
  non_zero_count: number;
}

interface EmbeddingResult {
  embedding: number[];
  dimension: number;
  model: string;
  version: string;
  timestamp: string;
  execution_time_ms: number;
  loading_time_ms: number;
  device_used: string;
  l2_norm: number;
  image_metadata: ImageMetadata;
  vector_stats: VectorStats;
  extra_metadata: Record<string, any>;
}

interface ModelInfo {
  name: string;
  version: string;
  task: string;
  status: string;
  device: string;
  dimension: number;
}

export default function EmbeddingsPage() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null);
  const [embeddingResult, setEmbeddingResult] = useState<EmbeddingResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [actionLoading, setActionLoading] = useState(false);

  // Fetch model info on mount
  useEffect(() => {
    fetchModelInfo();
  }, []);

  const fetchModelInfo = async () => {
    try {
      const res = await fetch("/api/v1/embeddings/model-info");
      const json = await res.json();
      if (json.success && json.data) {
        setModelInfo(json.data);
      }
    } catch (err) {
      console.error("Failed to fetch model info:", err);
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      setSelectedFile(file);
      setError(null);
      const reader = new FileReader();
      reader.onloadend = () => {
        setImagePreview(reader.result as string);
      };
      reader.readAsDataURL(file);
    }
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const file = e.dataTransfer.files[0];
      if (file.type.startsWith("image/")) {
        setSelectedFile(file);
        setError(null);
        const reader = new FileReader();
        reader.onloadend = () => {
          setImagePreview(reader.result as string);
        };
        reader.readAsDataURL(file);
      } else {
        setError("Please drop a valid image file (JPEG, PNG, WEBP).");
      }
    }
  };

  const handleGenerateEmbedding = async () => {
    if (!selectedFile) {
      setError("Please select or drop an image file first.");
      return;
    }

    setLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
      const res = await fetch("/api/v1/embeddings/generate", {
        method: "POST",
        body: formData,
      });

      const json = await res.json();
      if (json.success && json.data) {
        setEmbeddingResult(json.data);
        fetchModelInfo(); // Refresh status
      } else {
        setError(json.error?.message || json.detail || "Failed to generate image embedding.");
      }
    } catch (err: any) {
      setError(`Network error generating embedding: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  const handleLoadModel = async () => {
    setActionLoading(true);
    try {
      await fetch("/api/v1/embeddings/model/load", { method: "POST" });
      await fetchModelInfo();
    } catch (err) {
      console.error(err);
    } finally {
      setActionLoading(false);
    }
  };

  const handleUnloadModel = async () => {
    setActionLoading(true);
    try {
      await fetch("/api/v1/embeddings/model/unload", { method: "POST" });
      await fetchModelInfo();
    } catch (err) {
      console.error(err);
    } finally {
      setActionLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      <PageHeader
        title="Image Embeddings"
        description="Generate and inspect 768-dimensional L2-normalized dense vector representations for computer vision."
        breadcrumbs={["VisionForge", "Embeddings"]}
        actions={
          <div className="flex items-center space-x-2">
            <Button
              variant="secondary"
              size="sm"
              icon={<RefreshCw className={`w-3.5 h-3.5 ${actionLoading ? "animate-spin" : ""}`} />}
              onClick={handleLoadModel}
              disabled={actionLoading}
            >
              Load Weights
            </Button>
            <Button
              variant="secondary"
              size="sm"
              icon={<Trash2 className="w-3.5 h-3.5 text-rose-400" />}
              onClick={handleUnloadModel}
              disabled={actionLoading}
            >
              Unload Memory
            </Button>
          </div>
        }
      />

      {error && (
        <div className="p-4 bg-rose-500/10 border border-rose-500/30 rounded-xl text-rose-400 text-sm flex items-center gap-3">
          <AlertCircle className="w-5 h-5 shrink-0" />
          <span>{error}</span>
        </div>
      )}

      {/* Main Grid Workspace */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left Column: Image Upload & Trigger (5 cols) */}
        <div className="lg:col-span-5 space-y-6">
          <Card className="p-5 flex flex-col space-y-4">
            <div className="flex items-center justify-between">
              <h3 className="text-sm font-semibold text-neutral-200 flex items-center gap-2">
                <FileImage className="w-4 h-4 text-blue-400" />
                Input Image Modality
              </h3>
              {selectedFile && (
                <span className="text-[11px] font-mono text-neutral-400 bg-neutral-900 px-2 py-0.5 rounded border border-white/10">
                  {(selectedFile.size / 1024).toFixed(1)} KB
                </span>
              )}
            </div>

            {/* Drop Zone */}
            <div
              onDragOver={(e) => e.preventDefault()}
              onDrop={handleDrop}
              className={`border-2 border-dashed rounded-xl p-6 text-center transition-all flex flex-col items-center justify-center cursor-pointer min-h-[220px] ${
                imagePreview
                  ? "border-blue-500/50 bg-blue-500/5"
                  : "border-white/10 hover:border-blue-500/40 bg-neutral-950/50 hover:bg-neutral-900/40"
              }`}
              onClick={() => document.getElementById("file-upload-input")?.click()}
            >
              <input
                id="file-upload-input"
                type="file"
                accept="image/jpeg,image/png,image/webp"
                className="hidden"
                onChange={handleFileChange}
              />

              {imagePreview ? (
                <div className="relative group w-full flex flex-col items-center">
                  <img
                    src={imagePreview}
                    alt="Preview"
                    className="max-h-48 rounded-lg object-contain border border-white/10 shadow-lg"
                  />
                  <div className="mt-3 text-xs text-neutral-400 group-hover:text-blue-400 transition-colors">
                    Click or drop to replace image
                  </div>
                </div>
              ) : (
                <div className="flex flex-col items-center space-y-2">
                  <div className="w-12 h-12 rounded-full bg-blue-500/10 border border-blue-500/20 flex items-center justify-center text-blue-400">
                    <Upload className="w-5 h-5" />
                  </div>
                  <div className="text-sm font-medium text-neutral-200">
                    Drop image file here or click to browse
                  </div>
                  <div className="text-xs text-neutral-500">
                    Supports JPEG, PNG, WebP up to 20MB
                  </div>
                </div>
              )}
            </div>

            {/* Generate Action Button */}
            <Button
              variant="primary"
              size="lg"
              className="w-full justify-center"
              icon={<Sparkles className={`w-4 h-4 ${loading ? "animate-spin" : ""}`} />}
              onClick={handleGenerateEmbedding}
              disabled={loading || !selectedFile}
            >
              {loading ? "Executing SigLIP Pipeline..." : "Generate Vector Embedding"}
            </Button>
          </Card>

          {/* Model Specification Card */}
          <Card className="p-5 space-y-4">
            <h3 className="text-sm font-semibold text-neutral-200 flex items-center gap-2">
              <Cpu className="w-4 h-4 text-emerald-400" />
              Vision Model Architecture
            </h3>

            <div className="grid grid-cols-2 gap-3 text-xs">
              <div className="p-3 bg-neutral-900/60 rounded-lg border border-white/5 space-y-1">
                <div className="text-neutral-500 font-medium">Model Name</div>
                <div className="font-mono text-neutral-200 font-semibold truncate">
                  {modelInfo?.name || "siglip-base-patch16-224"}
                </div>
              </div>

              <div className="p-3 bg-neutral-900/60 rounded-lg border border-white/5 space-y-1">
                <div className="text-neutral-500 font-medium">Status</div>
                <div className="flex items-center gap-1.5 mt-0.5">
                  <Badge
                    variant={
                      modelInfo?.status === "ready"
                        ? "success"
                        : modelInfo?.status === "unloaded"
                        ? "neutral"
                        : "warning"
                    }
                  >
                    {modelInfo?.status || "ready"}
                  </Badge>
                </div>
              </div>

              <div className="p-3 bg-neutral-900/60 rounded-lg border border-white/5 space-y-1">
                <div className="text-neutral-500 font-medium">Vector Dimension</div>
                <div className="font-mono text-blue-400 font-semibold text-sm">
                  {modelInfo?.dimension || 768}D
                </div>
              </div>

              <div className="p-3 bg-neutral-900/60 rounded-lg border border-white/5 space-y-1">
                <div className="text-neutral-500 font-medium">Device Accelerator</div>
                <div className="font-mono text-emerald-400 font-semibold uppercase">
                  {modelInfo?.device || "cpu"}
                </div>
              </div>
            </div>
          </Card>
        </div>

        {/* Right Column: Telemetry & Vector Visualization (7 cols) */}
        <div className="lg:col-span-7 space-y-6">
          {embeddingResult ? (
            <>
              {/* Telemetry Summary Cards */}
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                <Card className="p-4 space-y-1 bg-neutral-900/40">
                  <div className="text-[11px] font-medium text-neutral-500 flex items-center gap-1">
                    <Zap className="w-3 h-3 text-amber-400" />
                    Inference Time
                  </div>
                  <div className="text-lg font-mono font-bold text-neutral-100">
                    {embeddingResult.execution_time_ms.toFixed(1)} <span className="text-xs text-neutral-500 font-sans">ms</span>
                  </div>
                </Card>

                <Card className="p-4 space-y-1 bg-neutral-900/40">
                  <div className="text-[11px] font-medium text-neutral-500 flex items-center gap-1">
                    <Activity className="w-3 h-3 text-blue-400" />
                    Load Duration
                  </div>
                  <div className="text-lg font-mono font-bold text-neutral-100">
                    {embeddingResult.loading_time_ms.toFixed(1)} <span className="text-xs text-neutral-500 font-sans">ms</span>
                  </div>
                </Card>

                <Card className="p-4 space-y-1 bg-neutral-900/40">
                  <div className="text-[11px] font-medium text-neutral-500 flex items-center gap-1">
                    <Hash className="w-3 h-3 text-emerald-400" />
                    L2 Norm
                  </div>
                  <div className="text-lg font-mono font-bold text-emerald-400">
                    {embeddingResult.l2_norm.toFixed(4)}
                  </div>
                </Card>

                <Card className="p-4 space-y-1 bg-neutral-900/40">
                  <div className="text-[11px] font-medium text-neutral-500 flex items-center gap-1">
                    <Layers className="w-3 h-3 text-purple-400" />
                    Dimension
                  </div>
                  <div className="text-lg font-mono font-bold text-purple-300">
                    {embeddingResult.dimension}
                  </div>
                </Card>
              </div>

              {/* Vector Statistical Summary & Sparkline Visualizer */}
              <Card className="p-5 space-y-5">
                <div className="flex items-center justify-between border-b border-white/10 pb-3">
                  <h3 className="text-sm font-semibold text-neutral-200 flex items-center gap-2">
                    <BarChart3 className="w-4 h-4 text-blue-400" />
                    Vector Component Statistics & Magnitude Distribution
                  </h3>
                  <Badge variant="info">768-D Dense Array</Badge>
                </div>

                {/* Magnitude Sparkline Bar Chart */}
                <div className="space-y-2">
                  <div className="text-xs text-neutral-400 font-medium flex justify-between">
                    <span>Component Magnitude Sparkline (Index 0..767)</span>
                    <span className="text-neutral-500 font-mono">Normalized L2 Unit Space</span>
                  </div>

                  <div className="h-28 bg-neutral-950 p-2.5 rounded-lg border border-white/5 flex items-end justify-between gap-[1px] overflow-hidden">
                    {embeddingResult.embedding.slice(0, 128).map((val, idx) => {
                      const heightPct = Math.min(Math.max((Math.abs(val) / (embeddingResult.vector_stats.max || 0.1)) * 100, 4), 100);
                      const isPositive = val >= 0;
                      return (
                        <div
                          key={idx}
                          title={`Index ${idx}: ${val.toFixed(4)}`}
                          style={{ height: `${heightPct}%` }}
                          className={`w-full rounded-t-[1px] transition-all hover:opacity-100 opacity-80 ${
                            isPositive ? "bg-blue-500 hover:bg-blue-400" : "bg-purple-500 hover:bg-purple-400"
                          }`}
                        />
                      );
                    })}
                  </div>
                </div>

                {/* Stats Table Grid */}
                <div className="grid grid-cols-2 sm:grid-cols-5 gap-3 pt-2">
                  <div className="p-2.5 bg-neutral-900/60 rounded-lg border border-white/5">
                    <div className="text-[10px] text-neutral-500 font-medium uppercase">Min Value</div>
                    <div className="font-mono text-xs text-neutral-200 font-semibold">
                      {embeddingResult.vector_stats.min.toFixed(4)}
                    </div>
                  </div>

                  <div className="p-2.5 bg-neutral-900/60 rounded-lg border border-white/5">
                    <div className="text-[10px] text-neutral-500 font-medium uppercase">Max Value</div>
                    <div className="font-mono text-xs text-neutral-200 font-semibold">
                      {embeddingResult.vector_stats.max.toFixed(4)}
                    </div>
                  </div>

                  <div className="p-2.5 bg-neutral-900/60 rounded-lg border border-white/5">
                    <div className="text-[10px] text-neutral-500 font-medium uppercase">Mean</div>
                    <div className="font-mono text-xs text-neutral-200 font-semibold">
                      {embeddingResult.vector_stats.mean.toFixed(4)}
                    </div>
                  </div>

                  <div className="p-2.5 bg-neutral-900/60 rounded-lg border border-white/5">
                    <div className="text-[10px] text-neutral-500 font-medium uppercase">Std Dev</div>
                    <div className="font-mono text-xs text-neutral-200 font-semibold">
                      {embeddingResult.vector_stats.std.toFixed(4)}
                    </div>
                  </div>

                  <div className="p-2.5 bg-neutral-900/60 rounded-lg border border-white/5">
                    <div className="text-[10px] text-neutral-500 font-medium uppercase">Non-Zero Count</div>
                    <div className="font-mono text-xs text-emerald-400 font-semibold">
                      {embeddingResult.vector_stats.non_zero_count} / {embeddingResult.dimension}
                    </div>
                  </div>
                </div>
              </Card>

              {/* Image Metadata Panel */}
              <Card className="p-5 space-y-3">
                <h3 className="text-sm font-semibold text-neutral-200 flex items-center gap-2">
                  <HardDrive className="w-4 h-4 text-neutral-400" />
                  Processed Image Metadata
                </h3>

                <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs">
                  <div className="p-2.5 bg-neutral-900/40 rounded-lg border border-white/5">
                    <div className="text-neutral-500">Dimensions</div>
                    <div className="font-mono text-neutral-200 font-medium">
                      {embeddingResult.image_metadata.width} x {embeddingResult.image_metadata.height}
                    </div>
                  </div>

                  <div className="p-2.5 bg-neutral-900/40 rounded-lg border border-white/5">
                    <div className="text-neutral-500">Format / Mode</div>
                    <div className="font-mono text-neutral-200 font-medium">
                      {embeddingResult.image_metadata.format} ({embeddingResult.image_metadata.mode})
                    </div>
                  </div>

                  <div className="p-2.5 bg-neutral-900/40 rounded-lg border border-white/5">
                    <div className="text-neutral-500">Aspect Ratio</div>
                    <div className="font-mono text-neutral-200 font-medium">
                      {embeddingResult.image_metadata.aspect_ratio}:1
                    </div>
                  </div>

                  <div className="p-2.5 bg-neutral-900/40 rounded-lg border border-white/5">
                    <div className="text-neutral-500">File Size</div>
                    <div className="font-mono text-neutral-200 font-medium">
                      {(embeddingResult.image_metadata.file_size_bytes / 1024).toFixed(1)} KB
                    </div>
                  </div>
                </div>
              </Card>
            </>
          ) : (
            <Card className="p-12 flex flex-col items-center justify-center text-center space-y-4 min-h-[420px] bg-neutral-900/20 border-dashed">
              <div className="w-14 h-14 rounded-2xl bg-blue-500/10 border border-blue-500/20 flex items-center justify-center text-blue-400">
                <BarChart3 className="w-7 h-7" />
              </div>
              <div className="space-y-1 max-w-sm">
                <h4 className="text-base font-semibold text-neutral-200">No Vector Generated Yet</h4>
                <p className="text-xs text-neutral-400">
                  Select or drop an image on the left panel and click &quot;Generate Vector Embedding&quot; to execute the SigLIP feature extraction pipeline.
                </p>
              </div>
            </Card>
          )}
        </div>
      </div>
    </div>
  );
}
