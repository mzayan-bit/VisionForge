"use client";

import React, { useEffect, useState, useRef, useCallback } from "react";
import { useRouter } from "next/navigation";
import {
  Compass,
  Sliders,
  Sparkles,
  Layers,
  Search,
  Zap,
  Info,
  ZoomIn,
  ZoomOut,
  RotateCcw,
  Tag,
  AlertTriangle,
  RefreshCw,
  Cpu,
  BarChart2,
  CheckCircle2,
  Filter,
} from "lucide-react";
import { PageHeader } from "@/components/ui/PageHeader";
import { Card } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";
import { Badge } from "@/components/ui/Badge";

interface ExplorerPoint {
  id: string;
  x: number;
  y: number;
  z?: number | null;
  image_metadata: Record<string, any>;
  tags: string[];
  embedding_model: string;
  cluster_id: number;
  outlier_score: number;
  distance_to_centroid: number;
}

interface DimensionalityReductionMeta {
  method: "pca" | "tsne";
  n_components: number;
  original_dimension: number;
  explained_variance_ratio: number[];
  cumulative_explained_variance: number;
  perplexity?: number | null;
  random_seed: number;
}

interface ClusteringMeta {
  method: string;
  n_clusters: number;
  cluster_sizes: Record<string, number>;
  inertia: number;
}

interface ExplorerDatasetPayload {
  dataset_id: string;
  timestamp: string;
  points: ExplorerPoint[];
  total_points: number;
  reduction_meta: DimensionalityReductionMeta;
  clustering_meta: ClusteringMeta;
  execution_time_ms: number;
  cached: boolean;
}

const CLUSTER_COLORS = [
  "#3b82f6", // Blue
  "#10b981", // Emerald
  "#a855f7", // Purple
  "#f59e0b", // Amber
  "#ec4899", // Pink
  "#06b6d4", // Cyan
  "#f97316", // Orange
  "#84cc16", // Lime
];

export default function EmbeddingExplorerPage() {
  const router = useRouter();

  // Control Form States
  const [method, setMethod] = useState<"pca" | "tsne">("pca");
  const [nComponents, setNComponents] = useState<2 | 3>(2);
  const [nClusters, setNClusters] = useState(3);
  const [perplexity, setPerplexity] = useState(30.0);
  const [randomSeed, setRandomSeed] = useState(42);

  // Payload & Loading States
  const [payload, setPayload] = useState<ExplorerDatasetPayload | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Selection & Filter States
  const [selectedPoint, setSelectedPoint] = useState<ExplorerPoint | null>(null);
  const [hoveredPoint, setHoveredPoint] = useState<ExplorerPoint | null>(null);
  const [clusterFilter, setClusterFilter] = useState<number | "all">("all");
  const [outliersOnly, setOutliersOnly] = useState(false);

  // Interactive Viewport Pan/Zoom States
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState(false);
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 });

  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  // Fetch initial stats & projection
  useEffect(() => {
    fetchProjection();
  }, []);

  const fetchProjection = async () => {
    setLoading(true);
    setError(null);

    try {
      const res = await fetch("/api/v1/explorer/project", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          method,
          n_components: nComponents,
          perplexity,
          random_seed: randomSeed,
          n_clusters: nClusters,
        }),
      });

      const json = await res.json();
      if (json.success && json.data) {
        setPayload(json.data);
        if (json.data.points.length > 0) {
          setSelectedPoint(json.data.points[0]);
        }
      } else {
        setError(json.detail || json.error?.message || "Failed to generate embedding projection.");
      }
    } catch (err: any) {
      setError(`Network error: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  // Filtered Points
  const getFilteredPoints = useCallback(() => {
    if (!payload) return [];
    return payload.points.filter((pt) => {
      if (clusterFilter !== "all" && pt.cluster_id !== clusterFilter) return false;
      if (outliersOnly && pt.outlier_score < 0.5) return false;
      return true;
    });
  }, [payload, clusterFilter, outliersOnly]);

  // Canvas Interactive Rendering Engine
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !payload) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const width = canvas.width;
    const height = canvas.height;

    // Clear Canvas
    ctx.clearRect(0, 0, width, height);

    // Render Grid Lines
    ctx.strokeStyle = "rgba(255, 255, 255, 0.04)";
    ctx.lineWidth = 1;
    const gridSize = 40 * zoom;
    for (let x = (pan.x % gridSize); x < width; x += gridSize) {
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, height);
      ctx.stroke();
    }
    for (let y = (pan.y % gridSize); y < height; y += gridSize) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(width, y);
      ctx.stroke();
    }

    const filtered = getFilteredPoints();
    if (filtered.length === 0) return;

    // Calculate Bounding Box for Normalization
    const xVals = payload.points.map((p) => p.x);
    const yVals = payload.points.map((p) => p.y);
    const minX = Math.min(...xVals);
    const maxX = Math.max(...xVals);
    const minY = Math.min(...yVals);
    const maxY = Math.max(...yVals);

    const rangeX = maxX - minX || 1.0;
    const rangeY = maxY - minY || 1.0;
    const margin = 60;

    // Map point (x, y) to Canvas coordinates (cx, cy)
    const mapToCanvas = (pt: ExplorerPoint) => {
      const normX = (pt.x - minX) / rangeX;
      const normY = (pt.y - minY) / rangeY;
      const cx = margin + normX * (width - 2 * margin) * zoom + pan.x;
      const cy = height - (margin + normY * (height - 2 * margin) * zoom) + pan.y;
      return { cx, cy };
    };

    // Render Connections / Points
    filtered.forEach((pt) => {
      const { cx, cy } = mapToCanvas(pt);
      const isSelected = selectedPoint?.id === pt.id;
      const isHovered = hoveredPoint?.id === pt.id;
      const isOutlier = pt.outlier_score >= 0.5;

      const baseColor = CLUSTER_COLORS[pt.cluster_id % CLUSTER_COLORS.length];

      // Draw Outlier Glow Ring if applicable
      if (isOutlier) {
        ctx.beginPath();
        ctx.arc(cx, cy, 14, 0, 2 * Math.PI);
        ctx.fillStyle = "rgba(244, 63, 94, 0.25)";
        ctx.fill();
        ctx.strokeStyle = "rgba(244, 63, 94, 0.8)";
        ctx.lineWidth = 1.5;
        ctx.stroke();
      }

      // Draw Selection Halo
      if (isSelected || isHovered) {
        ctx.beginPath();
        ctx.arc(cx, cy, 12, 0, 2 * Math.PI);
        ctx.strokeStyle = isSelected ? "#3b82f6" : "rgba(255, 255, 255, 0.6)";
        ctx.lineWidth = 2;
        ctx.stroke();
      }

      // Draw Main Point Node
      ctx.beginPath();
      ctx.arc(cx, cy, isSelected ? 7 : 5, 0, 2 * Math.PI);
      ctx.fillStyle = baseColor;
      ctx.fill();
      ctx.strokeStyle = "#000";
      ctx.lineWidth = 1;
      ctx.stroke();
    });
  }, [payload, selectedPoint, hoveredPoint, zoom, pan, getFilteredPoints]);

  // Handle Canvas Click & Hover Interactions
  const handleCanvasClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas || !payload) return;

    const rect = canvas.getBoundingClientRect();
    const clickX = e.clientX - rect.left;
    const clickY = e.clientY - rect.top;

    const filtered = getFilteredPoints();
    const xVals = payload.points.map((p) => p.x);
    const yVals = payload.points.map((p) => p.y);
    const minX = Math.min(...xVals);
    const maxX = Math.max(...xVals);
    const minY = Math.min(...yVals);
    const maxY = Math.max(...yVals);
    const rangeX = maxX - minX || 1.0;
    const rangeY = maxY - minY || 1.0;
    const margin = 60;

    let clicked: ExplorerPoint | null = null;
    let minDist = 20;

    filtered.forEach((pt) => {
      const normX = (pt.x - minX) / rangeX;
      const normY = (pt.y - minY) / rangeY;
      const cx = margin + normX * (canvas.width - 2 * margin) * zoom + pan.x;
      const cy = canvas.height - (margin + normY * (canvas.height - 2 * margin) * zoom) + pan.y;

      const dist = Math.hypot(clickX - cx, clickY - cy);
      if (dist < minDist) {
        minDist = dist;
        clicked = pt;
      }
    });

    if (clicked) {
      setSelectedPoint(clicked);
    }
  };

  // Drag Pan Controls
  const handleMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
    setIsDragging(true);
    setDragStart({ x: e.clientX - pan.x, y: e.clientY - pan.y });
  };

  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (isDragging) {
      setPan({ x: e.clientX - dragStart.x, y: e.clientY - dragStart.y });
    }
  };

  const handleMouseUp = () => setIsDragging(false);

  const resetViewport = () => {
    setZoom(1);
    setPan({ x: 0, y: 0 });
  };

  const triggerVisualSearch = (recordId: string) => {
    router.push(`/search?record_id=${encodeURIComponent(recordId)}`);
  };

  return (
    <div className="space-y-6">
      <PageHeader
        title="Embedding Explorer"
        description="Interactive 2D/3D visual space projection, K-Means clustering, and anomaly outlier detection for high-dimensional image embeddings."
        breadcrumbs={["VisionForge", "Embedding Explorer"]}
      />

      {error && (
        <div className="p-4 bg-rose-500/10 border border-rose-500/30 rounded-xl text-rose-400 text-sm flex items-center gap-3">
          <AlertTriangle className="w-5 h-5 shrink-0" />
          <span>{error}</span>
        </div>
      )}

      {/* Controls & Hyperparameter Form Card */}
      <Card className="p-6 space-y-5">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-white/10 pb-4">
          <div className="flex items-center space-x-2">
            <Button
              variant={method === "pca" ? "primary" : "secondary"}
              size="sm"
              icon={<Compass className="w-3.5 h-3.5" />}
              onClick={() => setMethod("pca")}
            >
              PCA (Linear Variance)
            </Button>
            <Button
              variant={method === "tsne" ? "primary" : "secondary"}
              size="sm"
              icon={<Sparkles className="w-3.5 h-3.5" />}
              onClick={() => setMethod("tsne")}
            >
              t-SNE (Non-Linear Manifold)
            </Button>
          </div>

          <div className="flex items-center space-x-3 text-xs font-mono">
            <span className="text-neutral-400">Model:</span>
            <Badge variant="info">siglip-base-patch16-224 (768D)</Badge>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-4 gap-4 text-xs">
          <div className="space-y-1.5 bg-neutral-950 p-3 rounded-lg border border-white/5">
            <div className="flex justify-between text-neutral-400 font-medium">
              <span>Projection Dimensions</span>
              <span className="font-mono text-blue-400 font-bold">{nComponents}D</span>
            </div>
            <div className="flex space-x-2 pt-1">
              <button
                type="button"
                className={`flex-1 py-1 rounded text-xs font-mono font-semibold transition-all ${
                  nComponents === 2 ? "bg-blue-500 text-white" : "bg-neutral-900 text-neutral-400"
                }`}
                onClick={() => setNComponents(2)}
              >
                2D Map
              </button>
              <button
                type="button"
                className={`flex-1 py-1 rounded text-xs font-mono font-semibold transition-all ${
                  nComponents === 3 ? "bg-blue-500 text-white" : "bg-neutral-900 text-neutral-400"
                }`}
                onClick={() => setNComponents(3)}
              >
                3D Map
              </button>
            </div>
          </div>

          <div className="space-y-1.5 bg-neutral-950 p-3 rounded-lg border border-white/5">
            <div className="flex justify-between text-neutral-400 font-medium">
              <span>K-Means Clusters</span>
              <span className="font-mono text-purple-400 font-bold">{nClusters}</span>
            </div>
            <input
              type="range"
              min={1}
              max={8}
              value={nClusters}
              onChange={(e) => setNClusters(parseInt(e.target.value))}
              className="w-full accent-purple-500 mt-2"
            />
          </div>

          {method === "tsne" ? (
            <div className="space-y-1.5 bg-neutral-950 p-3 rounded-lg border border-white/5">
              <div className="flex justify-between text-neutral-400 font-medium">
                <span>t-SNE Perplexity</span>
                <span className="font-mono text-emerald-400 font-bold">{perplexity}</span>
              </div>
              <input
                type="range"
                min={5}
                max={50}
                value={perplexity}
                onChange={(e) => setPerplexity(parseFloat(e.target.value))}
                className="w-full accent-emerald-500 mt-2"
              />
            </div>
          ) : (
            <div className="space-y-1.5 bg-neutral-950 p-3 rounded-lg border border-white/5">
              <div className="flex justify-between text-neutral-400 font-medium">
                <span>Random Seed</span>
                <span className="font-mono text-amber-400 font-bold">{randomSeed}</span>
              </div>
              <input
                type="number"
                value={randomSeed}
                onChange={(e) => setRandomSeed(parseInt(e.target.value) || 42)}
                className="w-full bg-neutral-900 border border-white/10 rounded px-2 py-1 text-xs text-neutral-200 focus:outline-none"
              />
            </div>
          )}

          <div className="flex items-end">
            <Button
              variant="primary"
              size="lg"
              className="w-full justify-center"
              icon={<RefreshCw className={`w-4 h-4 ${loading ? "animate-spin" : ""}`} />}
              onClick={fetchProjection}
              disabled={loading}
            >
              {loading ? "Computing Projection..." : "Generate Projection"}
            </Button>
          </div>
        </div>
      </Card>

      {/* Telemetry & Explained Variance Bar */}
      {payload && (
        <Card className="p-4 bg-neutral-900/50 space-y-3 border-blue-500/20">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 text-xs border-b border-white/5 pb-2">
            <div className="flex items-center space-x-3 font-mono">
              <span className="text-neutral-400">Dataset ID:</span>
              <span className="text-blue-400 font-semibold">{payload.dataset_id}</span>
              {payload.cached ? (
                <Badge variant="info">Cached Payload</Badge>
              ) : (
                <Badge variant="success">Fresh Computed</Badge>
              )}
            </div>

            <div className="flex items-center space-x-3 text-neutral-400 font-mono">
              <span>Timestamp: {new Date(payload.timestamp).toLocaleTimeString()}</span>
            </div>
          </div>

          <div className="grid grid-cols-2 sm:grid-cols-5 gap-3 text-xs font-mono">
            <div className="p-2.5 bg-neutral-950 rounded-lg border border-white/5">
              <div className="text-neutral-500 text-[10px]">Projected Points</div>
              <div className="text-base text-neutral-100 font-bold">{payload.total_points}</div>
            </div>

            <div className="p-2.5 bg-neutral-950 rounded-lg border border-white/5">
              <div className="text-neutral-500 text-[10px]">Original Dim</div>
              <div className="text-base text-purple-400 font-bold">{payload.reduction_meta.original_dimension}D</div>
            </div>

            <div className="p-2.5 bg-neutral-950 rounded-lg border border-white/5">
              <div className="text-neutral-500 text-[10px]">K-Means Inertia</div>
              <div className="text-base text-emerald-400 font-bold">{payload.clustering_meta.inertia.toFixed(1)}</div>
            </div>

            <div className="p-2.5 bg-neutral-950 rounded-lg border border-white/5">
              <div className="text-neutral-500 text-[10px]">Explained Variance</div>
              <div className="text-base text-amber-400 font-bold">
                {payload.reduction_meta.method === "pca"
                  ? `${(payload.reduction_meta.cumulative_explained_variance * 100).toFixed(1)}%`
                  : "N/A (Non-linear)"}
              </div>
            </div>

            <div className="p-2.5 bg-neutral-950 rounded-lg border border-white/5">
              <div className="text-neutral-500 text-[10px]">Execution Latency</div>
              <div className="text-base text-blue-400 font-bold">{payload.execution_time_ms.toFixed(1)}ms</div>
            </div>
          </div>
        </Card>
      )}

      {/* Main Interactive Visual Map Workspace */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left: Interactive Map Canvas (8 cols) */}
        <div className="lg:col-span-8 space-y-4">
          <Card className="p-4 space-y-3 bg-neutral-950/80 border-white/10 relative">
            <div className="flex items-center justify-between border-b border-white/10 pb-3">
              <div className="flex items-center space-x-3">
                <span className="text-xs font-semibold text-neutral-200 uppercase tracking-wider flex items-center gap-2">
                  <BarChart2 className="w-4 h-4 text-blue-400" />
                  Embedding Space Projection Map ({nComponents}D)
                </span>
              </div>

              <div className="flex items-center space-x-2">
                <button
                  onClick={() => setZoom((z) => Math.min(z * 1.2, 5))}
                  className="p-1.5 rounded bg-neutral-900 hover:bg-neutral-800 text-neutral-300 transition-colors"
                  title="Zoom In"
                >
                  <ZoomIn className="w-3.5 h-3.5" />
                </button>
                <button
                  onClick={() => setZoom((z) => Math.max(z / 1.2, 0.5))}
                  className="p-1.5 rounded bg-neutral-900 hover:bg-neutral-800 text-neutral-300 transition-colors"
                  title="Zoom Out"
                >
                  <ZoomOut className="w-3.5 h-3.5" />
                </button>
                <button
                  onClick={resetViewport}
                  className="p-1.5 rounded bg-neutral-900 hover:bg-neutral-800 text-neutral-300 transition-colors"
                  title="Reset Viewport"
                >
                  <RotateCcw className="w-3.5 h-3.5" />
                </button>
              </div>
            </div>

            {/* Filter Bar */}
            <div className="flex flex-wrap items-center justify-between gap-3 text-xs pt-1 pb-2">
              <div className="flex items-center space-x-2">
                <Filter className="w-3.5 h-3.5 text-neutral-500" />
                <span className="text-neutral-400">Cluster Filter:</span>
                <select
                  value={clusterFilter}
                  onChange={(e) => setClusterFilter(e.target.value === "all" ? "all" : parseInt(e.target.value))}
                  className="bg-neutral-900 border border-white/10 rounded px-2.5 py-1 text-xs text-neutral-200 focus:outline-none"
                >
                  <option value="all">All Clusters</option>
                  {Array.from({ length: nClusters }).map((_, idx) => (
                    <option key={idx} value={idx}>
                      Cluster #{idx}
                    </option>
                  ))}
                </select>
              </div>

              <label className="flex items-center space-x-2 cursor-pointer text-neutral-300 select-none">
                <input
                  type="checkbox"
                  checked={outliersOnly}
                  onChange={(e) => setOutliersOnly(e.target.checked)}
                  className="rounded accent-rose-500"
                />
                <span>Show Anomaly Outliers Only (&gt; 0.5)</span>
              </label>
            </div>

            {/* Canvas Container */}
            <div className="relative w-full overflow-hidden rounded-xl bg-[#09090b] border border-white/5 min-h-[460px] flex items-center justify-center">
              {payload && payload.total_points > 0 ? (
                <canvas
                  ref={canvasRef}
                  width={720}
                  height={460}
                  className="w-full h-[460px] cursor-grab active:cursor-grabbing"
                  onClick={handleCanvasClick}
                  onMouseDown={handleMouseDown}
                  onMouseMove={handleMouseMove}
                  onMouseUp={handleMouseUp}
                />
              ) : (
                <div className="py-20 text-center space-y-3">
                  <div className="w-12 h-12 rounded-full bg-neutral-900 text-neutral-500 mx-auto flex items-center justify-center border border-white/10">
                    <Compass className="w-6 h-6" />
                  </div>
                  <div className="text-xs text-neutral-400 max-w-xs mx-auto">
                    No indexed items in Visual Memory. Upload images in Visual Search or Indexing to populate embedding space.
                  </div>
                </div>
              )}
            </div>

            {/* Cluster Legend */}
            <div className="flex flex-wrap items-center gap-3 pt-2 text-xs font-mono">
              <span className="text-neutral-500">Clusters:</span>
              {Array.from({ length: nClusters }).map((_, idx) => (
                <div key={idx} className="flex items-center space-x-1.5">
                  <span
                    className="w-2.5 h-2.5 rounded-full"
                    style={{ backgroundColor: CLUSTER_COLORS[idx % CLUSTER_COLORS.length] }}
                  />
                  <span className="text-neutral-300">Cluster #{idx}</span>
                </div>
              ))}
            </div>
          </Card>
        </div>

        {/* Right: Selected Point Inspector (4 cols) */}
        <div className="lg:col-span-4 space-y-4">
          <Card className="p-5 space-y-4">
            <div className="text-xs font-semibold text-neutral-300 uppercase tracking-wider flex items-center gap-2 border-b border-white/10 pb-3">
              <Zap className="w-4 h-4 text-amber-400" />
              Selected Point Inspector
            </div>

            {selectedPoint ? (
              <div className="space-y-4">
                <div className="p-3 bg-neutral-950 rounded-xl border border-white/10 space-y-2">
                  <div className="flex items-center justify-between">
                    <span className="font-mono text-xs font-bold text-neutral-100 truncate max-w-[180px]">
                      {selectedPoint.id}
                    </span>
                    <Badge
                      style={{
                        backgroundColor: `${CLUSTER_COLORS[selectedPoint.cluster_id % CLUSTER_COLORS.length]}20`,
                        color: CLUSTER_COLORS[selectedPoint.cluster_id % CLUSTER_COLORS.length],
                        borderColor: `${CLUSTER_COLORS[selectedPoint.cluster_id % CLUSTER_COLORS.length]}40`,
                      }}
                    >
                      Cluster #{selectedPoint.cluster_id}
                    </Badge>
                  </div>

                  <div className="grid grid-cols-2 gap-2 text-[11px] font-mono text-neutral-400 bg-neutral-900/60 p-2.5 rounded-lg">
                    <div>Coord X: <span className="text-neutral-200">{selectedPoint.x.toFixed(3)}</span></div>
                    <div>Coord Y: <span className="text-neutral-200">{selectedPoint.y.toFixed(3)}</span></div>
                    {selectedPoint.z !== null && (
                      <div>Coord Z: <span className="text-neutral-200">{selectedPoint.z?.toFixed(3)}</span></div>
                    )}
                    <div>Distance: <span className="text-neutral-200">{selectedPoint.distance_to_centroid.toFixed(3)}</span></div>
                  </div>
                </div>

                <div className="p-3 bg-neutral-950 rounded-xl border border-white/10 space-y-2 text-xs font-mono">
                  <div className="flex items-center justify-between border-b border-white/5 pb-1.5">
                    <span className="text-neutral-400">Outlier Anomaly Score</span>
                    <span
                      className={`font-bold ${
                        selectedPoint.outlier_score >= 0.5 ? "text-rose-400" : "text-emerald-400"
                      }`}
                    >
                      {(selectedPoint.outlier_score * 100).toFixed(1)}%
                    </span>
                  </div>

                  <div className="w-full bg-neutral-900 h-2 rounded-full overflow-hidden">
                    <div
                      className={`h-full transition-all ${
                        selectedPoint.outlier_score >= 0.5 ? "bg-rose-500" : "bg-emerald-500"
                      }`}
                      style={{ width: `${selectedPoint.outlier_score * 100}%` }}
                    />
                  </div>
                </div>

                <div className="p-3 bg-neutral-950 rounded-xl border border-white/10 space-y-2 text-xs font-mono">
                  <div className="text-neutral-400 font-semibold border-b border-white/5 pb-1">Image Metadata</div>
                  <div className="space-y-1 text-neutral-300 text-[11px]">
                    <div>Resolution: {selectedPoint.image_metadata?.width || 224} x {selectedPoint.image_metadata?.height || 224} px</div>
                    <div>Format: {selectedPoint.image_metadata?.format || "JPEG/PNG"}</div>
                    <div>Model: {selectedPoint.embedding_model}</div>
                  </div>

                  {selectedPoint.tags.length > 0 && (
                    <div className="flex flex-wrap gap-1 pt-1">
                      {selectedPoint.tags.map((t) => (
                        <span key={t} className="text-[10px] bg-blue-500/10 text-blue-300 px-2 py-0.5 rounded border border-blue-500/20 flex items-center gap-1">
                          <Tag className="w-2.5 h-2.5" />
                          {t}
                        </span>
                      ))}
                    </div>
                  )}
                </div>

                <Button
                  variant="primary"
                  size="md"
                  className="w-full justify-center"
                  icon={<Search className="w-3.5 h-3.5" />}
                  onClick={() => triggerVisualSearch(selectedPoint.id)}
                >
                  Find Visually Similar Images
                </Button>
              </div>
            ) : (
              <div className="py-12 text-center text-xs text-neutral-500 space-y-2 border border-dashed border-white/10 rounded-xl">
                <Info className="w-6 h-6 mx-auto text-neutral-600" />
                <div>Click any point on the projection map to inspect details.</div>
              </div>
            )}
          </Card>
        </div>
      </div>
    </div>
  );
}
