"use client";

import React, { useEffect, useState } from "react";
import {
  Search,
  Database,
  Upload,
  Sparkles,
  Layers,
  HardDrive,
  Trash2,
  AlertCircle,
  Clock,
  Tag,
  CheckCircle2,
  Sliders,
  Image as ImageIcon,
} from "lucide-react";
import { PageHeader } from "@/components/ui/PageHeader";
import { Card } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";
import { Badge } from "@/components/ui/Badge";

interface MemoryStats {
  total_records: number;
  vector_dimension: number;
  memory_size_mb: number;
  storage_path: string;
  last_saved_at: string | null;
}

interface SearchResultItem {
  id: string;
  similarity_score: number;
  distance: number;
  image_metadata: Record<string, any>;
  tags: string[];
  indexed_at: string;
}

interface SearchResponse {
  results: SearchResultItem[];
  query_execution_time_ms: number;
  candidate_count: number;
  metric_used: string;
}

export default function VisualSearchPage() {
  // Indexing State
  const [indexFile, setIndexFile] = useState<File | null>(null);
  const [indexPreview, setIndexPreview] = useState<string | null>(null);
  const [customTags, setCustomTags] = useState("");
  const [indexing, setIndexing] = useState(false);
  const [indexSuccess, setIndexSuccess] = useState<string | null>(null);

  // Search State
  const [queryFile, setQueryFile] = useState<File | null>(null);
  const [queryPreview, setQueryPreview] = useState<string | null>(null);
  const [topK, setTopK] = useState(5);
  const [metric, setMetric] = useState("cosine");
  const [threshold, setThreshold] = useState(0.0);
  const [searching, setSearching] = useState(false);

  // Stats & Results State
  const [stats, setStats] = useState<MemoryStats | null>(null);
  const [searchResults, setSearchResults] = useState<SearchResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchMemoryStats();
  }, []);

  const fetchMemoryStats = async () => {
    try {
      const res = await fetch("/api/v1/memory/stats");
      const json = await res.json();
      if (json.success && json.data) {
        setStats(json.data);
      }
    } catch (err) {
      console.error("Failed to fetch memory stats:", err);
    }
  };

  const handleIndexFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      setIndexFile(file);
      const reader = new FileReader();
      reader.onloadend = () => setIndexPreview(reader.result as string);
      reader.readAsDataURL(file);
    }
  };

  const handleQueryFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      setQueryFile(file);
      const reader = new FileReader();
      reader.onloadend = () => setQueryPreview(reader.result as string);
      reader.readAsDataURL(file);
    }
  };

  const handleIndexSubmit = async () => {
    if (!indexFile) return;
    setIndexing(true);
    setError(null);
    setIndexSuccess(null);

    const formData = new FormData();
    formData.append("file", indexFile);
    if (customTags) formData.append("tags", customTags);

    try {
      const res = await fetch("/api/v1/memory/index", {
        method: "POST",
        body: formData,
      });
      const json = await res.json();
      if (json.success && json.data) {
        setIndexSuccess(`Indexed image successfully as '${json.data.id}'`);
        setIndexFile(null);
        setIndexPreview(null);
        setCustomTags("");
        fetchMemoryStats();
      } else {
        setError(json.detail || json.error?.message || "Failed to index image.");
      }
    } catch (err: any) {
      setError(`Network error indexing image: ${err.message}`);
    } finally {
      setIndexing(false);
    }
  };

  const handleSearchSubmit = async () => {
    if (!queryFile) return;
    setSearching(true);
    setError(null);

    const formData = new FormData();
    formData.append("file", queryFile);
    formData.append("top_k", topK.toString());
    formData.append("metric", metric);
    formData.append("threshold", threshold.toString());

    try {
      const res = await fetch("/api/v1/search/image", {
        method: "POST",
        body: formData,
      });
      const json = await res.json();
      if (json.success && json.data) {
        setSearchResults(json.data);
      } else {
        setError(json.detail || json.error?.message || "Visual search failed.");
      }
    } catch (err: any) {
      setError(`Network error searching visual memory: ${err.message}`);
    } finally {
      setSearching(false);
    }
  };

  const handleClearMemory = async () => {
    if (!confirm("Are you sure you want to clear all indexed visual memory?")) return;
    try {
      await fetch("/api/v1/memory/clear", { method: "DELETE" });
      setSearchResults(null);
      fetchMemoryStats();
    } catch (err) {
      console.error("Failed to clear visual memory:", err);
    }
  };

  return (
    <div className="space-y-6">
      <PageHeader
        title="Visual Search & Memory"
        description="Search and query indexed visual memories using dense 768D vector similarity matching."
        breadcrumbs={["VisionForge", "Visual Search"]}
        actions={
          <Button
            variant="secondary"
            size="sm"
            icon={<Trash2 className="w-3.5 h-3.5 text-rose-400" />}
            onClick={handleClearMemory}
          >
            Clear Visual Memory
          </Button>
        }
      />

      {error && (
        <div className="p-4 bg-rose-500/10 border border-rose-500/30 rounded-xl text-rose-400 text-sm flex items-center gap-3">
          <AlertCircle className="w-5 h-5 shrink-0" />
          <span>{error}</span>
        </div>
      )}

      {indexSuccess && (
        <div className="p-4 bg-emerald-500/10 border border-emerald-500/30 rounded-xl text-emerald-400 text-sm flex items-center gap-3">
          <CheckCircle2 className="w-5 h-5 shrink-0" />
          <span>{indexSuccess}</span>
        </div>
      )}

      {/* Telemetry Bar */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
        <Card className="p-4 space-y-1 bg-neutral-900/40">
          <div className="text-xs text-neutral-500 font-medium flex items-center gap-1.5">
            <Database className="w-3.5 h-3.5 text-blue-400" />
            Total Memory Vectors
          </div>
          <div className="text-xl font-mono font-bold text-neutral-100">
            {stats?.total_records ?? 0}
          </div>
        </Card>

        <Card className="p-4 space-y-1 bg-neutral-900/40">
          <div className="text-xs text-neutral-500 font-medium flex items-center gap-1.5">
            <Layers className="w-3.5 h-3.5 text-purple-400" />
            Vector Dimension
          </div>
          <div className="text-xl font-mono font-bold text-purple-300">
            {stats?.vector_dimension ?? 768}D
          </div>
        </Card>

        <Card className="p-4 space-y-1 bg-neutral-900/40">
          <div className="text-xs text-neutral-500 font-medium flex items-center gap-1.5">
            <HardDrive className="w-3.5 h-3.5 text-emerald-400" />
            Memory Size
          </div>
          <div className="text-xl font-mono font-bold text-emerald-400">
            {stats?.memory_size_mb.toFixed(3) ?? "0.000"} <span className="text-xs text-neutral-500 font-sans">MB</span>
          </div>
        </Card>

        <Card className="p-4 space-y-1 bg-neutral-900/40">
          <div className="text-xs text-neutral-500 font-medium flex items-center gap-1.5">
            <Clock className="w-3.5 h-3.5 text-amber-400" />
            Last Disk Sync
          </div>
          <div className="text-xs font-mono text-neutral-300 truncate mt-1">
            {stats?.last_saved_at ? new Date(stats.last_saved_at).toLocaleTimeString() : "Never"}
          </div>
        </Card>
      </div>

      {/* Main Grid Workspace */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left Column: Indexing & Search Controls (5 cols) */}
        <div className="lg:col-span-5 space-y-6">
          {/* Section 1: Index New Image */}
          <Card className="p-5 space-y-4">
            <h3 className="text-sm font-semibold text-neutral-200 flex items-center gap-2 border-b border-white/10 pb-3">
              <Upload className="w-4 h-4 text-emerald-400" />
              1. Index Image into Visual Memory
            </h3>

            <div
              className="border-2 border-dashed border-white/10 rounded-xl p-4 text-center cursor-pointer hover:border-emerald-500/40 bg-neutral-950/40 transition-all min-h-[120px] flex flex-col items-center justify-center"
              onClick={() => document.getElementById("index-file-input")?.click()}
            >
              <input
                id="index-file-input"
                type="file"
                accept="image/*"
                className="hidden"
                onChange={handleIndexFileChange}
              />
              {indexPreview ? (
                <img src={indexPreview} alt="Index Preview" className="max-h-24 rounded border border-white/10 object-contain" />
              ) : (
                <div className="text-xs text-neutral-400">Click to select image to index</div>
              )}
            </div>

            <div className="space-y-1">
              <label className="text-xs text-neutral-400 font-medium">Custom Tags (Optional)</label>
              <input
                type="text"
                placeholder="e.g. nature, portrait, landscape"
                value={customTags}
                onChange={(e) => setCustomTags(e.target.value)}
                className="w-full bg-neutral-950 border border-white/10 rounded-lg px-3 py-1.5 text-xs text-neutral-200 focus:outline-none focus:border-emerald-500/50"
              />
            </div>

            <Button
              variant="secondary"
              className="w-full justify-center"
              icon={<Database className={`w-3.5 h-3.5 ${indexing ? "animate-spin" : ""}`} />}
              onClick={handleIndexSubmit}
              disabled={indexing || !indexFile}
            >
              {indexing ? "Indexing Vector..." : "Add to Visual Memory"}
            </Button>
          </Card>

          {/* Section 2: Visual Search Query */}
          <Card className="p-5 space-y-4">
            <h3 className="text-sm font-semibold text-neutral-200 flex items-center gap-2 border-b border-white/10 pb-3">
              <Search className="w-4 h-4 text-blue-400" />
              2. Query Visual Search Engine
            </h3>

            <div
              className="border-2 border-dashed border-white/10 rounded-xl p-4 text-center cursor-pointer hover:border-blue-500/40 bg-neutral-950/40 transition-all min-h-[120px] flex flex-col items-center justify-center"
              onClick={() => document.getElementById("query-file-input")?.click()}
            >
              <input
                id="query-file-input"
                type="file"
                accept="image/*"
                className="hidden"
                onChange={handleQueryFileChange}
              />
              {queryPreview ? (
                <img src={queryPreview} alt="Query Preview" className="max-h-24 rounded border border-white/10 object-contain" />
              ) : (
                <div className="text-xs text-neutral-400">Click to select query image</div>
              )}
            </div>

            {/* Controls */}
            <div className="space-y-3 text-xs">
              <div className="flex justify-between items-center">
                <label className="text-neutral-400 font-medium">Distance Metric</label>
                <select
                  value={metric}
                  onChange={(e) => setMetric(e.target.value)}
                  className="bg-neutral-950 border border-white/10 rounded px-2 py-1 text-xs text-neutral-200 focus:outline-none"
                >
                  <option value="cosine">Cosine Similarity</option>
                  <option value="dot_product">Dot Product</option>
                  <option value="euclidean">Euclidean Distance</option>
                </select>
              </div>

              <div className="space-y-1">
                <div className="flex justify-between text-neutral-400">
                  <span>Top-K Matches</span>
                  <span className="font-mono text-blue-400 font-semibold">{topK}</span>
                </div>
                <input
                  type="range"
                  min={1}
                  max={20}
                  value={topK}
                  onChange={(e) => setTopK(parseInt(e.target.value))}
                  className="w-full accent-blue-500"
                />
              </div>

              <div className="space-y-1">
                <div className="flex justify-between text-neutral-400">
                  <span>Minimum Threshold</span>
                  <span className="font-mono text-emerald-400 font-semibold">{(threshold * 100).toFixed(0)}%</span>
                </div>
                <input
                  type="range"
                  min={0.0}
                  max={0.95}
                  step={0.05}
                  value={threshold}
                  onChange={(e) => setThreshold(parseFloat(e.target.value))}
                  className="w-full accent-emerald-500"
                />
              </div>
            </div>

            <Button
              variant="primary"
              size="lg"
              className="w-full justify-center"
              icon={<Sparkles className={`w-4 h-4 ${searching ? "animate-spin" : ""}`} />}
              onClick={handleSearchSubmit}
              disabled={searching || !queryFile}
            >
              {searching ? "Searching Memory..." : "Execute Visual Search"}
            </Button>
          </Card>
        </div>

        {/* Right Column: Search Results Gallery & Inspector (7 cols) */}
        <div className="lg:col-span-7 space-y-6">
          {searchResults ? (
            <Card className="p-5 space-y-5">
              <div className="flex items-center justify-between border-b border-white/10 pb-3">
                <h3 className="text-sm font-semibold text-neutral-200 flex items-center gap-2">
                  <Sliders className="w-4 h-4 text-blue-400" />
                  Ranked Visual Matches ({searchResults.results.length})
                </h3>
                <div className="flex items-center gap-2 text-xs font-mono text-neutral-400">
                  <span>Duration: {searchResults.query_execution_time_ms.toFixed(1)}ms</span>
                  <span>•</span>
                  <span>Candidates: {searchResults.candidate_count}</span>
                </div>
              </div>

              {searchResults.results.length === 0 ? (
                <div className="py-12 text-center text-xs text-neutral-400">
                  No visual memory matches satisfied threshold ({(threshold * 100).toFixed(0)}%).
                </div>
              ) : (
                <div className="space-y-3">
                  {searchResults.results.map((item, index) => (
                    <div
                      key={item.id}
                      className="p-4 bg-neutral-900/60 border border-white/5 rounded-xl flex items-center justify-between hover:border-blue-500/30 transition-all group"
                    >
                      <div className="flex items-center space-x-4">
                        <div className="w-9 h-9 rounded-lg bg-blue-500/10 border border-blue-500/20 flex items-center justify-center font-mono text-sm font-bold text-blue-400">
                          #{index + 1}
                        </div>

                        <div className="space-y-1">
                          <div className="flex items-center space-x-2">
                            <span className="font-mono text-xs text-neutral-200 font-semibold">{item.id}</span>
                            {item.tags.map((t) => (
                              <span key={t} className="text-[10px] bg-neutral-800 text-neutral-400 px-1.5 py-0.5 rounded flex items-center gap-1">
                                <Tag className="w-2.5 h-2.5" />
                                {t}
                              </span>
                            ))}
                          </div>
                          <div className="text-[11px] text-neutral-500 font-mono">
                            Distance: {item.distance.toFixed(4)} • Indexed: {new Date(item.indexed_at).toLocaleTimeString()}
                          </div>
                        </div>
                      </div>

                      <div className="text-right">
                        <div className="text-lg font-mono font-bold text-emerald-400">
                          {(item.similarity_score * 100).toFixed(1)}%
                        </div>
                        <div className="text-[10px] text-neutral-500 uppercase">Match Score</div>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </Card>
          ) : (
            <Card className="p-12 flex flex-col items-center justify-center text-center space-y-4 min-h-[440px] bg-neutral-900/20 border-dashed">
              <div className="w-14 h-14 rounded-2xl bg-blue-500/10 border border-blue-500/20 flex items-center justify-center text-blue-400">
                <Search className="w-7 h-7" />
              </div>
              <div className="space-y-1 max-w-sm">
                <h4 className="text-base font-semibold text-neutral-200">Visual Search Ready</h4>
                <p className="text-xs text-neutral-400">
                  Index images into visual memory on the left, then upload a query image to perform vectorized similarity search across all indexed memories.
                </p>
              </div>
            </Card>
          )}
        </div>
      </div>
    </div>
  );
}
