"use client";

import React, { useEffect, useState } from "react";
import Link from "next/link";
import {
  Activity,
  AlertCircle,
  ArrowRight,
  BarChart3,
  CheckCircle2,
  Clock,
  Compass,
  Copy,
  Cpu,
  Database,
  Download,
  Eye,
  FileImage,
  Filter,
  HardDrive,
  History,
  Info,
  Layers,
  MapPin,
  Move,
  Play,
  RefreshCw,
  Search,
  Sliders,
  Sparkles,
  Tag,
  Upload,
  Video,
  X,
  Zap,
} from "lucide-react";
import { PageHeader } from "@/components/ui/PageHeader";
import { Card } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";
import { Badge } from "@/components/ui/Badge";

// ─── Interfaces ───────────────────────────────────────────────────

interface VisualAsset {
  asset_id: string;
  asset_type: "IMAGE" | "FRAME" | "OBJECT_CROP" | "DATASET_SAMPLE" | "EVENT_FRAME";
  title: string;
  embedding_id: string;
  embedding_model: string;
  embedding_version: string;
  source_video_id?: string;
  source_dataset_id?: string;
  source_run_id?: string;
  source_event_id?: string;
  timestamp_sec?: number;
  frame_idx?: number;
  track_id?: number;
  bbox?: number[];
  class_name?: string;
  thumbnail_url?: string;
  metadata: Record<string, any>;
  indexed_at: string;
}

interface UnifiedSearchResultItem {
  rank: number;
  asset: VisualAsset;
  similarity_score: number;
  distance: number;
  source_traceability: Record<string, any>;
  action_link: string;
  evidence_notes: string;
}

interface UnifiedSearchResponse {
  search_id: string;
  timestamp: string;
  query_summary: string;
  query_asset?: VisualAsset;
  results: UnifiedSearchResultItem[];
  candidate_count: number;
  returned_count: number;
  metric_used: string;
  model_used: string;
  embedding_time_ms: number;
  search_time_ms: number;
  filtering_time_ms: number;
  total_execution_time_ms: number;
  explanation: string;
}

interface NearDuplicatePair {
  asset_a: VisualAsset;
  asset_b: VisualAsset;
  similarity_score: number;
  distance: number;
  recommendation: string;
}

interface NearDuplicateResponse {
  total_evaluated: number;
  duplicate_pairs_found: number;
  pairs: NearDuplicatePair[];
  threshold_used: number;
  execution_time_ms: number;
}

interface SearchHistoryRecord {
  search_id: string;
  timestamp: string;
  query_type: string;
  query_info: Record<string, any>;
  model_used: string;
  top_k: number;
  threshold: number;
  metric_used: string;
  candidate_count: number;
  returned_count: number;
  total_time_ms: number;
}

export default function UnifiedVisualSearchPage() {
  // Search Mode: "image" | "frame" | "object" | "event" | "duplicates"
  const [searchMode, setSearchMode] = useState<"image" | "frame" | "object" | "event" | "duplicates">("image");

  // Query Inputs State
  const [queryFile, setQueryFile] = useState<File | null>(null);
  const [queryPreview, setQueryPreview] = useState<string | null>(null);
  const [queryVideoId, setQueryVideoId] = useState<string>("sample_traffic_01");
  const [queryTimestampSec, setQueryTimestampSec] = useState<number>(4.0);
  const [queryTrackId, setQueryTrackId] = useState<number>(1);
  const [queryEventId, setQueryEventId] = useState<string>("evt_dwell_01");

  // Search Filters & Hyperparameters
  const [filterAssetType, setFilterAssetType] = useState<string>("ALL");
  const [filterClass, setFilterClass] = useState<string>("ALL");
  const [topK, setTopK] = useState<number>(10);
  const [metric, setMetric] = useState<string>("cosine");
  const [threshold, setThreshold] = useState<number>(0.0);
  const [duplicateThreshold, setDuplicateThreshold] = useState<number>(0.95);

  // Results & Inspector State
  const [searchResults, setSearchResults] = useState<UnifiedSearchResponse | null>(null);
  const [duplicateResults, setDuplicateResults] = useState<NearDuplicateResponse | null>(null);
  const [inspectItem, setInspectItem] = useState<UnifiedSearchResultItem | null>(null);
  const [searchHistory, setSearchHistory] = useState<SearchHistoryRecord[]>([]);
  const [showHistoryDrawer, setShowHistoryDrawer] = useState<boolean>(false);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchSearchHistory();
    // Pre-populate index by running default search
    handleRunSearch();
  }, []);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setQueryFile(file);
      const reader = new FileReader();
      reader.onloadend = () => {
        setQueryPreview(reader.result as string);
      };
      reader.readAsDataURL(file);
    }
  };

  const handleRunSearch = async () => {
    setLoading(true);
    setError(null);
    try {
      if (searchMode === "duplicates") {
        const res = await fetch(`/api/v1/search/duplicates?threshold=${duplicateThreshold}`);
        const json = await res.json();
        if (json.success) {
          setDuplicateResults(json.data);
        } else {
          setError(json.message || "Failed to find near duplicates");
        }
      } else if (searchMode === "image" && queryFile) {
        const formData = new FormData();
        formData.append("file", queryFile);
        formData.append("top_k", topK.toString());
        formData.append("metric", metric);
        formData.append("threshold", threshold.toString());

        const res = await fetch("/api/v1/search/image", {
          method: "POST",
          body: formData,
        });
        const json = await res.json();
        if (json.success) {
          // Map to UnifiedSearchResponse structure
          const oldPayload = json.data;
          const mapped: UnifiedSearchResponse = {
            search_id: oldPayload.search_id,
            timestamp: oldPayload.timestamp,
            query_summary: `Uploaded Image (${queryFile.name})`,
            results: oldPayload.results.map((r: any) => ({
              rank: r.rank,
              asset: {
                asset_id: `asset_${r.id}`,
                asset_type: "IMAGE",
                title: r.image_metadata?.title || `Visual Memory Item ${r.id.slice(0, 8)}`,
                embedding_id: r.id,
                embedding_model: r.embedding_model,
                embedding_version: "1.0.0",
                metadata: r.image_metadata,
                indexed_at: r.indexed_at,
              },
              similarity_score: r.similarity_score,
              distance: r.distance,
              source_traceability: { record_id: r.id, tags: r.tags },
              action_link: "/search",
              evidence_notes: `Matched visual memory item ${r.id.slice(0, 8)} (similarity: ${(r.similarity_score * 100).toFixed(1)}%)`,
            })),
            candidate_count: oldPayload.candidate_count,
            returned_count: oldPayload.returned_count,
            metric_used: oldPayload.metric_used,
            model_used: oldPayload.model_used,
            embedding_time_ms: oldPayload.embedding_time_ms,
            search_time_ms: oldPayload.search_time_ms,
            filtering_time_ms: 0.0,
            total_execution_time_ms: oldPayload.total_execution_time_ms,
            explanation: `Ranked ${oldPayload.returned_count} matches by dense embedding similarity (SigLIP-base-patch16-224). Metric: ${metric.toUpperCase()}.`,
          };
          setSearchResults(mapped);
          if (mapped.results.length > 0) setInspectItem(mapped.results[0]);
        }
      } else {
        // Unified Search API
        const payload: Record<string, any> = {
          query_type:
            searchMode === "frame"
              ? "FRAME"
              : searchMode === "object"
              ? "OBJECT_CROP"
              : searchMode === "event"
              ? "EVENT_FRAME"
              : "IMAGE",
          video_id: queryVideoId,
          timestamp_sec: queryTimestampSec,
          run_id: "vrun_query_test_01",
          track_id: queryTrackId,
          event_id: queryEventId,
          top_k: topK,
          threshold: threshold,
          metric: metric,
          filter_asset_types: filterAssetType === "ALL" ? null : [filterAssetType],
          filter_class_name: filterClass === "ALL" ? null : filterClass,
        };

        const res = await fetch("/api/v1/search/unified", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        });
        const json = await res.json();
        if (json.success) {
          setSearchResults(json.data);
          if (json.data.results.length > 0) {
            setInspectItem(json.data.results[0]);
          }
        } else {
          setError(json.message || "Unified visual search failed");
        }
      }
      await fetchSearchHistory();
    } catch (err: any) {
      setError(err.message || "Visual search execution error");
    } finally {
      setLoading(false);
    }
  };

  const fetchSearchHistory = async () => {
    try {
      const res = await fetch("/api/v1/search/history?limit=30");
      const json = await res.json();
      if (json.success && Array.isArray(json.data)) {
        setSearchHistory(json.data);
      }
    } catch (err) {
      console.error("Failed to fetch search history:", err);
    }
  };

  return (
    <div className="flex flex-col min-h-screen bg-[#0a0a0a] text-neutral-200 font-inter">
      {/* Page Header */}
      <PageHeader
        title="Unified Visual Search"
        description="Search across Images, Video Frames, Track Object Crops, Moments, and Dataset Samples using Dense Embedding Similarity (SigLIP)"
        breadcrumbs={["VisionForge", "Visual Search"]}
        actions={
          <div className="flex items-center gap-2">
            <Button
              variant="secondary"
              icon={<History className="w-4 h-4 text-purple-400" />}
              onClick={() => setShowHistoryDrawer(true)}
            >
              Search History ({searchHistory.length})
            </Button>

            <Link href="/explorer">
              <Button variant="secondary" icon={<Compass className="w-4 h-4 text-blue-400" />}>
                Embedding Explorer
              </Button>
            </Link>
          </div>
        }
      />

      <div className="p-6 space-y-6 flex-1">
        {/* Search Mode Navigation Tabs */}
        <div className="flex flex-wrap items-center gap-2 border-b border-white/10 pb-4">
          {[
            { id: "image", label: "Image Upload", icon: <FileImage className="w-4 h-4" /> },
            { id: "frame", label: "Video Frame", icon: <Video className="w-4 h-4" /> },
            { id: "object", label: "Object Crop", icon: <Layers className="w-4 h-4" /> },
            { id: "event", label: "Event Moment", icon: <Activity className="w-4 h-4" /> },
            { id: "duplicates", label: "Near-Duplicate Candidates", icon: <Copy className="w-4 h-4 text-amber-400" /> },
          ].map((tab) => (
            <button
              key={tab.id}
              onClick={() => setSearchMode(tab.id as any)}
              className={`flex items-center gap-2 px-4 py-2 rounded-xl font-mono text-xs font-semibold transition-all ${
                searchMode === tab.id
                  ? "bg-blue-600 text-white shadow-lg shadow-blue-600/30"
                  : "bg-[#141414] hover:bg-[#1c1c1c] text-neutral-400 border border-white/5"
              }`}
            >
              {tab.icon}
              {tab.label}
            </button>
          ))}
        </div>

        {/* Query Input & Filters Section */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Left Column: Query Target Input */}
          <div className="lg:col-span-2 bg-[#121212] border border-white/10 rounded-xl p-5 space-y-4 shadow-xl">
            <h3 className="text-xs font-semibold uppercase tracking-wider text-blue-400 flex items-center gap-2 border-b border-white/10 pb-3">
              <Search className="w-4 h-4 text-blue-400" />
              Query Input Specification ({searchMode.toUpperCase()})
            </h3>

            {/* Mode 1: Image Upload */}
            {searchMode === "image" && (
              <div className="space-y-4">
                <div className="border-2 border-dashed border-white/15 rounded-xl p-6 text-center hover:border-blue-500/50 transition-all bg-[#161616]">
                  {queryPreview ? (
                    <div className="space-y-3">
                      <img
                        src={queryPreview}
                        alt="Query Preview"
                        className="max-h-48 mx-auto rounded-lg shadow object-contain border border-white/10"
                      />
                      <button
                        onClick={() => {
                          setQueryFile(null);
                          setQueryPreview(null);
                        }}
                        className="text-xs text-red-400 hover:underline font-mono"
                      >
                        Remove Image
                      </button>
                    </div>
                  ) : (
                    <label className="cursor-pointer space-y-2 block">
                      <Upload className="w-8 h-8 text-neutral-500 mx-auto" />
                      <div className="text-xs font-mono text-neutral-300">
                        Drop query image here, or <span className="text-blue-400">browse file</span>
                      </div>
                      <div className="text-[10px] text-neutral-500 font-mono">
                        PNG, JPG, WebP up to 20MB
                      </div>
                      <input
                        type="file"
                        accept="image/*"
                        onChange={handleFileChange}
                        className="hidden"
                      />
                    </label>
                  )}
                </div>
              </div>
            )}

            {/* Mode 2: Video Frame */}
            {searchMode === "frame" && (
              <div className="space-y-4 font-mono text-xs">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <label className="text-neutral-400 block mb-1.5">Video Asset</label>
                    <select
                      value={queryVideoId}
                      onChange={(e) => setQueryVideoId(e.target.value)}
                      className="w-full bg-[#181818] border border-white/10 rounded-lg p-2.5 text-white"
                    >
                      <option value="sample_traffic_01">sample_traffic_01.mp4 (Traffic intersection)</option>
                      <option value="factory_safety_02">factory_safety_02.mp4 (Industrial plant)</option>
                    </select>
                  </div>

                  <div>
                    <label className="text-neutral-400 block mb-1.5">
                      Timestamp: {queryTimestampSec.toFixed(1)}s (Frame #{Math.round(queryTimestampSec * 30)})
                    </label>
                    <input
                      type="range"
                      min={0}
                      max={15}
                      step={0.5}
                      value={queryTimestampSec}
                      onChange={(e) => setQueryTimestampSec(parseFloat(e.target.value))}
                      className="w-full h-2 bg-[#1f1f1f] rounded-lg appearance-none cursor-pointer accent-blue-500 mt-2"
                    />
                  </div>
                </div>

                <div className="p-3 bg-[#181818] rounded-lg border border-white/5 text-[11px] text-neutral-400">
                  Target Query: Video frame sampled at <span className="text-blue-400 font-bold">t = {queryTimestampSec.toFixed(1)}s</span> from <span className="text-white font-bold">{queryVideoId}</span>.
                </div>
              </div>
            )}

            {/* Mode 3: Object Crop */}
            {searchMode === "object" && (
              <div className="space-y-4 font-mono text-xs">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <label className="text-neutral-400 block mb-1.5">Source Video Run</label>
                    <input
                      type="text"
                      disabled
                      value="vrun_query_test_01 (sample_traffic_01.mp4)"
                      className="w-full bg-[#181818] border border-white/10 rounded-lg p-2.5 text-neutral-400"
                    />
                  </div>

                  <div>
                    <label className="text-neutral-400 block mb-1.5">Target Track ID</label>
                    <select
                      value={queryTrackId}
                      onChange={(e) => setQueryTrackId(parseInt(e.target.value))}
                      className="w-full bg-[#181818] border border-white/10 rounded-lg p-2.5 text-white"
                    >
                      <option value={1}>Track #1 (person — 10.0s duration)</option>
                      <option value={2}>Track #2 (car — 8.5s duration)</option>
                      <option value={4}>Track #4 (person — 11.0s duration)</option>
                    </select>
                  </div>
                </div>

                <div className="p-3 bg-[#181818] rounded-lg border border-white/5 text-[11px] text-neutral-400">
                  Target Query: Visual appearance crop of <span className="text-purple-400 font-bold">Track #{queryTrackId}</span> extracted from representative detection bounding box.
                </div>
              </div>
            )}

            {/* Mode 4: Event Moment */}
            {searchMode === "event" && (
              <div className="space-y-4 font-mono text-xs">
                <div>
                  <label className="text-neutral-400 block mb-1.5">Target Temporal Event</label>
                  <select
                    value={queryEventId}
                    onChange={(e) => setQueryEventId(e.target.value)}
                    className="w-full bg-[#181818] border border-white/10 rounded-lg p-2.5 text-white"
                  >
                    <option value="evt_dwell_01">OBJECT_DWELLED (Track #4 in Loading Zone A, 4.0s)</option>
                    <option value="evt_entered_02">OBJECT_ENTERED_REGION (Track #1 into Zone A)</option>
                    <option value="evt_close_03">OBJECTS_BECAME_CLOSE (Track #1 & Track #2, 45.2px)</option>
                  </select>
                </div>

                <div className="p-3 bg-[#181818] rounded-lg border border-white/5 text-[11px] text-neutral-400">
                  Target Query: Keyframe evidence snapshot extracted from <span className="text-emerald-400 font-bold">{queryEventId}</span>.
                </div>
              </div>
            )}

            {/* Mode 5: Near-Duplicates */}
            {searchMode === "duplicates" && (
              <div className="space-y-4 font-mono text-xs">
                <div>
                  <label className="text-neutral-400 block mb-1.5">
                    Duplicate Similarity Cutoff: {(duplicateThreshold * 100).toFixed(0)}%
                  </label>
                  <input
                    type="range"
                    min={0.85}
                    max={0.99}
                    step={0.01}
                    value={duplicateThreshold}
                    onChange={(e) => setDuplicateThreshold(parseFloat(e.target.value))}
                    className="w-full h-2 bg-[#1f1f1f] rounded-lg appearance-none cursor-pointer accent-amber-500"
                  />
                </div>

                <div className="p-3 bg-[#181818] rounded-lg border border-white/5 text-[11px] text-neutral-400">
                  Discovers candidate duplicate pairs with pairwise cosine similarity $\ge$ {(duplicateThreshold * 100).toFixed(0)}%.
                </div>
              </div>
            )}

            <div className="flex justify-end pt-2">
              <Button
                variant="primary"
                icon={<Zap className={`w-4 h-4 ${loading ? "animate-spin" : ""}`} />}
                onClick={handleRunSearch}
                disabled={loading || (searchMode === "image" && !queryFile)}
              >
                {loading ? "Searching..." : searchMode === "duplicates" ? "Discover Duplicates" : "Find Visually Similar"}
              </Button>
            </div>
          </div>

          {/* Right Column: Search Filters & Hyperparameters */}
          <div className="bg-[#121212] border border-white/10 rounded-xl p-5 space-y-4 font-mono text-xs shadow-xl">
            <h3 className="font-semibold uppercase tracking-wider text-neutral-400 flex items-center gap-2 border-b border-white/10 pb-3">
              <Sliders className="w-4 h-4 text-amber-400" />
              Search Filters & Thresholds
            </h3>

            <div>
              <label className="text-neutral-400 block mb-1">Filter Asset Type</label>
              <select
                value={filterAssetType}
                onChange={(e) => setFilterAssetType(e.target.value)}
                className="w-full bg-[#181818] border border-white/10 rounded-lg p-2 text-white"
              >
                <option value="ALL">All Asset Types</option>
                <option value="IMAGE">Image Samples</option>
                <option value="FRAME">Video Frames</option>
                <option value="OBJECT_CROP">Object Crops</option>
                <option value="EVENT_FRAME">Event Evidence Frames</option>
                <option value="DATASET_SAMPLE">Dataset Samples</option>
              </select>
            </div>

            <div>
              <label className="text-neutral-400 block mb-1">Class Filter</label>
              <select
                value={filterClass}
                onChange={(e) => setFilterClass(e.target.value)}
                className="w-full bg-[#181818] border border-white/10 rounded-lg p-2 text-white"
              >
                <option value="ALL">All Classes</option>
                <option value="person">person</option>
                <option value="car">car</option>
                <option value="helmet">helmet</option>
              </select>
            </div>

            <div>
              <label className="text-neutral-400 block mb-1">
                Min Similarity Cutoff: {(threshold * 100).toFixed(0)}%
              </label>
              <input
                type="range"
                min={0.0}
                max={0.9}
                step={0.05}
                value={threshold}
                onChange={(e) => setThreshold(parseFloat(e.target.value))}
                className="w-full h-1.5 bg-[#1f1f1f] rounded-lg appearance-none cursor-pointer accent-blue-500"
              />
            </div>

            <div className="grid grid-cols-2 gap-3">
              <div>
                <label className="text-neutral-400 block mb-1">Top-K</label>
                <select
                  value={topK}
                  onChange={(e) => setTopK(parseInt(e.target.value))}
                  className="w-full bg-[#181818] border border-white/10 rounded-lg p-2 text-white"
                >
                  <option value={5}>Top 5</option>
                  <option value={10}>Top 10</option>
                  <option value={25}>Top 25</option>
                  <option value={50}>Top 50</option>
                </select>
              </div>

              <div>
                <label className="text-neutral-400 block mb-1">Metric</label>
                <select
                  value={metric}
                  onChange={(e) => setMetric(e.target.value)}
                  className="w-full bg-[#181818] border border-white/10 rounded-lg p-2 text-white"
                >
                  <option value="cosine">Cosine</option>
                  <option value="euclidean">Euclidean</option>
                </select>
              </div>
            </div>
          </div>
        </div>

        {/* Error Alert */}
        {error && (
          <div className="p-4 bg-red-950/30 border border-red-500/40 rounded-xl text-red-300 font-mono text-xs flex items-center gap-2">
            <AlertCircle className="w-4 h-4 text-red-400 shrink-0" />
            {error}
          </div>
        )}

        {/* Results Section */}
        {searchMode === "duplicates" && duplicateResults && (
          <div className="space-y-4">
            <div className="flex justify-between items-center bg-[#121212] border border-white/10 rounded-xl p-4 font-mono text-xs">
              <span className="text-neutral-300 font-bold">
                Near-Duplicate Candidates ({duplicateResults.duplicate_pairs_found} pairs identified in {duplicateResults.execution_time_ms}ms)
              </span>
              <span className="text-amber-400">
                Threshold $\ge$ {(duplicateResults.threshold_used * 100).toFixed(0)}%
              </span>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {duplicateResults.pairs.map((pair, idx) => (
                <div
                  key={idx}
                  className="bg-[#121212] border border-amber-500/30 rounded-xl p-4 space-y-3 font-mono text-xs"
                >
                  <div className="flex justify-between items-center border-b border-white/10 pb-2">
                    <span className="text-amber-400 font-bold">
                      Pair #{idx + 1}: {(pair.similarity_score * 100).toFixed(1)}% Cosine Similarity
                    </span>
                    <span className="text-neutral-500 text-[10px]">Distance: {pair.distance.toFixed(3)}</span>
                  </div>

                  <div className="grid grid-cols-2 gap-3 text-[11px]">
                    <div className="p-2.5 bg-[#181818] rounded border border-white/5 space-y-1">
                      <div className="text-blue-400 font-bold">{pair.asset_a.title}</div>
                      <div className="text-neutral-500 text-[10px]">{pair.asset_a.asset_type}</div>
                    </div>

                    <div className="p-2.5 bg-[#181818] rounded border border-white/5 space-y-1">
                      <div className="text-purple-400 font-bold">{pair.asset_b.title}</div>
                      <div className="text-neutral-500 text-[10px]">{pair.asset_b.asset_type}</div>
                    </div>
                  </div>

                  <div className="text-[10px] text-neutral-400 bg-[#161616] p-2 rounded border border-white/5">
                    💡 {pair.recommendation}
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {searchMode !== "duplicates" && searchResults && (
          <div className="space-y-6">
            {/* Telemetry Bar */}
            <div className="bg-[#121212] border border-white/10 rounded-xl p-4 flex flex-wrap justify-between items-center gap-4 font-mono text-xs">
              <div className="flex items-center gap-3">
                <span className="text-emerald-400 font-bold">
                  {searchResults.returned_count} matches found
                </span>
                <span className="text-neutral-500">|</span>
                <span className="text-neutral-400">
                  Evaluated {searchResults.candidate_count} indexed visual assets
                </span>
              </div>

              <div className="flex items-center gap-4 text-neutral-500 text-[11px]">
                <span>Search: {searchResults.search_time_ms}ms</span>
                <span>Total: {searchResults.total_execution_time_ms}ms</span>
                <span className="text-blue-400">{searchResults.model_used}</span>
              </div>
            </div>

            {/* Results Grid & Detail Inspector */}
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              {/* Left 2 Cols: Results Cards Grid */}
              <div className="lg:col-span-2 grid grid-cols-1 md:grid-cols-2 gap-4">
                {searchResults.results.map((item) => {
                  const isSelected = inspectItem?.asset.asset_id === item.asset.asset_id;
                  const simPct = (item.similarity_score * 100).toFixed(1);

                  return (
                    <div
                      key={item.asset.asset_id}
                      onClick={() => setInspectItem(item)}
                      className={`bg-[#121212] border rounded-xl p-4 space-y-3 cursor-pointer transition-all ${
                        isSelected
                          ? "border-blue-500 bg-blue-950/10 shadow-lg shadow-blue-500/20"
                          : "border-white/10 hover:border-white/20"
                      }`}
                    >
                      <div className="flex justify-between items-center font-mono text-xs">
                        <span
                          className={`px-2 py-0.5 rounded font-bold ${
                            item.similarity_score > 0.9
                              ? "bg-emerald-500/20 text-emerald-400 border border-emerald-500/30"
                              : item.similarity_score > 0.8
                              ? "bg-blue-500/20 text-blue-400 border border-blue-500/30"
                              : "bg-neutral-800 text-neutral-300 border border-white/10"
                          }`}
                        >
                          {simPct}% Similarity
                        </span>

                        <span className="text-[10px] text-neutral-500 font-mono">
                          Rank #{item.rank}
                        </span>
                      </div>

                      {/* Visual Card Canvas Box */}
                      <div className="aspect-video bg-[#0c0c0c] rounded-lg border border-white/5 flex flex-col items-center justify-center p-3 relative overflow-hidden">
                        <div className="absolute inset-0 bg-[radial-gradient(#1f1f1f_1px,transparent_1px)] [background-size:12px_12px] opacity-40" />
                        <div className="z-10 text-center space-y-1 font-mono">
                          <div className="text-xs font-bold text-white">{item.asset.title}</div>
                          <span className="text-[10px] px-1.5 py-0.5 rounded bg-blue-600/30 text-blue-300 font-bold">
                            {item.asset.asset_type}
                          </span>
                        </div>
                      </div>

                      <div className="space-y-1 font-mono text-[11px] text-neutral-400">
                        <div className="flex justify-between">
                          <span>Source:</span>
                          <span className="text-white">
                            {item.asset.source_video_id || item.asset.source_dataset_id || "Visual Memory"}
                          </span>
                        </div>
                        {item.asset.timestamp_sec !== undefined && (
                          <div className="flex justify-between">
                            <span>Timestamp:</span>
                            <span className="text-cyan-400">{item.asset.timestamp_sec.toFixed(1)}s</span>
                          </div>
                        )}
                        {item.asset.track_id !== undefined && (
                          <div className="flex justify-between">
                            <span>Track:</span>
                            <span className="text-purple-400">#{item.asset.track_id}</span>
                          </div>
                        )}
                      </div>

                      {/* Action Links */}
                      <div className="flex items-center gap-2 pt-2 border-t border-white/5 font-mono text-[10px]">
                        <Link href={item.action_link} className="flex-1">
                          <button className="w-full text-center py-1 rounded bg-blue-600/20 hover:bg-blue-600/30 text-blue-300 border border-blue-500/30 transition-all font-bold">
                            [ Open Source ]
                          </button>
                        </Link>

                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            setSearchMode("frame");
                            if (item.asset.source_video_id) setQueryVideoId(item.asset.source_video_id);
                            if (item.asset.timestamp_sec !== undefined) setQueryTimestampSec(item.asset.timestamp_sec);
                            handleRunSearch();
                          }}
                          className="px-2 py-1 rounded bg-[#1c1c1c] hover:bg-[#252525] text-neutral-300 border border-white/5"
                          title="Search similar to this match"
                        >
                          Find Similar
                        </button>
                      </div>
                    </div>
                  );
                })}
              </div>

              {/* Right Col: Selected Result Deep Inspector */}
              <div className="space-y-4">
                {inspectItem ? (
                  <div className="bg-[#121212] border border-white/10 rounded-xl p-5 space-y-4 font-mono text-xs sticky top-6">
                    <div className="flex justify-between items-center border-b border-white/10 pb-3">
                      <h4 className="font-semibold text-white flex items-center gap-2">
                        <Eye className="w-4 h-4 text-blue-400" />
                        Selected Match Inspector
                      </h4>
                      <span className="text-emerald-400 font-bold">
                        {(inspectItem.similarity_score * 100).toFixed(1)}% Match
                      </span>
                    </div>

                    <div className="space-y-2">
                      <div className="text-neutral-400 text-[10px]">Asset Title</div>
                      <div className="font-bold text-white text-sm">{inspectItem.asset.title}</div>
                    </div>

                    <div className="grid grid-cols-2 gap-3 text-[11px]">
                      <div className="p-2.5 bg-[#181818] rounded border border-white/5 space-y-0.5">
                        <div className="text-neutral-500 text-[10px]">Asset Type</div>
                        <div className="font-bold text-blue-400">{inspectItem.asset.asset_type}</div>
                      </div>

                      <div className="p-2.5 bg-[#181818] rounded border border-white/5 space-y-0.5">
                        <div className="text-neutral-500 text-[10px]">Distance</div>
                        <div className="font-bold text-white">{inspectItem.distance.toFixed(4)}</div>
                      </div>

                      <div className="p-2.5 bg-[#181818] rounded border border-white/5 space-y-0.5">
                        <div className="text-neutral-500 text-[10px]">Model Space</div>
                        <div className="font-bold text-amber-400">{inspectItem.asset.embedding_model}</div>
                      </div>

                      <div className="p-2.5 bg-[#181818] rounded border border-white/5 space-y-0.5">
                        <div className="text-neutral-500 text-[10px]">Dimension</div>
                        <div className="font-bold text-purple-400">768 Dense Float32</div>
                      </div>
                    </div>

                    <div className="p-3 bg-[#161616] rounded-lg border border-white/5 space-y-1.5">
                      <div className="text-neutral-400 font-bold text-[10px] uppercase tracking-wider">
                        Source Provenance Trace
                      </div>
                      <div className="text-neutral-300 text-[11px] leading-relaxed">
                        {inspectItem.evidence_notes}
                      </div>
                    </div>

                    <div className="pt-2 flex flex-col gap-2">
                      <Link href={inspectItem.action_link}>
                        <Button variant="primary" size="sm" className="w-full" icon={<Video className="w-3.5 h-3.5" />}>
                          Open in Video Lab
                        </Button>
                      </Link>

                      <Link href={`/explorer`}>
                        <Button variant="secondary" size="sm" className="w-full" icon={<Compass className="w-3.5 h-3.5" />}>
                          View in Embedding Explorer
                        </Button>
                      </Link>
                    </div>
                  </div>
                ) : (
                  <div className="p-8 text-center text-xs text-neutral-500 bg-[#121212] border border-white/10 rounded-xl font-mono">
                    Select any search result card to inspect vector telemetry and source provenance.
                  </div>
                )}
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Search History Drawer */}
      {showHistoryDrawer && (
        <div className="fixed inset-0 bg-black/80 z-50 flex items-center justify-end">
          <div className="bg-[#121212] border-l border-white/10 w-full max-w-md h-full p-6 space-y-4 font-mono text-xs overflow-y-auto">
            <div className="flex justify-between items-center border-b border-white/10 pb-3">
              <h3 className="text-sm font-semibold text-white flex items-center gap-2">
                <History className="w-4 h-4 text-purple-400" />
                Visual Search History Logs
              </h3>
              <button onClick={() => setShowHistoryDrawer(false)} className="text-neutral-400 hover:text-white">
                ✕
              </button>
            </div>

            <div className="space-y-2">
              {searchHistory.map((h) => (
                <div
                  key={h.search_id}
                  className="p-3 bg-[#181818] border border-white/5 rounded-lg space-y-1 hover:border-purple-500/40 transition-all"
                >
                  <div className="flex justify-between items-center">
                    <span className="text-purple-400 font-bold">{h.query_type}</span>
                    <span className="text-[10px] text-neutral-500">{h.total_time_ms}ms</span>
                  </div>
                  <div className="text-white text-xs font-semibold">
                    {h.query_info?.summary || `Search ${h.search_id}`}
                  </div>
                  <div className="flex justify-between items-center text-[10px] text-neutral-500 pt-1">
                    <span>Matches: {h.returned_count}</span>
                    <span>{new Date(h.timestamp).toLocaleTimeString()}</span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
