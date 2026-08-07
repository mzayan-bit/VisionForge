"use client";

import React, { useEffect, useState } from "react";
import {
  Search,
  Database,
  Upload,
  Sparkles,
  Layers,
  HardDrive,
  Clock,
  Tag,
  CheckCircle2,
  AlertCircle,
  Sliders,
  FileImage,
  Info,
  X,
  Zap,
  Cpu,
  BarChart3,
  List,
} from "lucide-react";
import { PageHeader } from "@/components/ui/PageHeader";
import { Card } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";
import { Badge } from "@/components/ui/Badge";

interface VisualMemoryRecord {
  id: string;
  embedding: number[];
  dimension: number;
  image_metadata: Record<string, any>;
  tags: string[];
  indexed_at: string;
}

interface SearchResultItem {
  rank: number;
  id: string;
  similarity_score: number;
  distance: number;
  image_metadata: Record<string, any>;
  tags: string[];
  indexed_at: string;
  embedding_model: string;
}

interface SearchResponse {
  search_id: string;
  timestamp: string;
  results: SearchResultItem[];
  candidate_count: number;
  returned_count: number;
  metric_used: string;
  model_used: string;
  embedding_time_ms: number;
  search_time_ms: number;
  total_execution_time_ms: number;
  query_info: Record<string, any>;
}

export default function VisualSearchPage() {
  // Query Input Mode: "upload" | "record"
  const [queryMode, setQueryMode] = useState<"upload" | "record">("upload");

  // Upload Query State
  const [queryFile, setQueryFile] = useState<File | null>(null);
  const [queryPreview, setQueryPreview] = useState<string | null>(null);

  // Selected Memory Record State
  const [memoryRecords, setMemoryRecords] = useState<VisualMemoryRecord[]>([]);
  const [selectedRecordId, setSelectedRecordId] = useState<string>("");

  // Search Settings State
  const [topK, setTopK] = useState(5);
  const [metric, setMetric] = useState("cosine");
  const [threshold, setThreshold] = useState(0.0);
  const [searching, setSearching] = useState(false);

  // Results & History State
  const [searchResults, setSearchResults] = useState<SearchResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [inspectItem, setInspectItem] = useState<SearchResultItem | null>(null);

  // Fetch Visual Memory records for record selection dropdown
  useEffect(() => {
    fetchMemoryRecords();
  }, []);

  const fetchMemoryRecords = async () => {
    try {
      const res = await fetch("/api/v1/memory/records?limit=100");
      const json = await res.json();
      if (json.success && Array.isArray(json.data)) {
        setMemoryRecords(json.data);
        if (json.data.length > 0 && !selectedRecordId) {
          setSelectedRecordId(json.data[0].id);
        }
      }
    } catch (err) {
      console.error("Failed to fetch visual memory records:", err);
    }
  };

  const handleQueryFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      setQueryFile(file);
      setError(null);
      const reader = new FileReader();
      reader.onloadend = () => setQueryPreview(reader.result as string);
      reader.readAsDataURL(file);
    }
  };

  const handleSearchExecute = async () => {
    setSearching(true);
    setError(null);

    try {
      let res: Response;
      if (queryMode === "upload") {
        if (!queryFile) {
          setError("Please select or drop a query image file.");
          setSearching(false);
          return;
        }

        const formData = new FormData();
        formData.append("file", queryFile);
        formData.append("top_k", topK.toString());
        formData.append("metric", metric);
        formData.append("threshold", threshold.toString());

        res = await fetch("/api/v1/search/image", {
          method: "POST",
          body: formData,
        });
      } else {
        if (!selectedRecordId) {
          setError("Please select an existing record from Visual Memory.");
          setSearching(false);
          return;
        }

        res = await fetch("/api/v1/search/record", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            record_id: selectedRecordId,
            top_k: topK,
            metric: metric,
            threshold: threshold,
          }),
        });
      }

      const json = await res.json();
      if (json.success && json.data) {
        setSearchResults(json.data);
      } else {
        setError(json.detail || json.error?.message || "Visual search failed.");
      }
    } catch (err: any) {
      setError(`Network error executing visual search: ${err.message}`);
    } finally {
      setSearching(false);
    }
  };

  return (
    <div className="space-y-6">
      <PageHeader
        title="Visual Search Workbench"
        description="Research-grade visual similarity search across indexed Visual Memory using dense 768D vector matching."
        breadcrumbs={["VisionForge", "Visual Search"]}
      />

      {error && (
        <div className="p-4 bg-rose-500/10 border border-rose-500/30 rounded-xl text-rose-400 text-sm flex items-center gap-3">
          <AlertCircle className="w-5 h-5 shrink-0" />
          <span>{error}</span>
        </div>
      )}

      {/* Query Input & Search Parameters Panel */}
      <Card className="p-6 space-y-6">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-white/10 pb-4">
          <div className="flex items-center space-x-2">
            <Button
              variant={queryMode === "upload" ? "primary" : "secondary"}
              size="sm"
              icon={<Upload className="w-3.5 h-3.5" />}
              onClick={() => setQueryMode("upload")}
            >
              Upload Query Image
            </Button>
            <Button
              variant={queryMode === "record" ? "primary" : "secondary"}
              size="sm"
              icon={<Database className="w-3.5 h-3.5" />}
              onClick={() => {
                setQueryMode("record");
                fetchMemoryRecords();
              }}
            >
              Select from Visual Memory ({memoryRecords.length})
            </Button>
          </div>

          <div className="flex items-center space-x-2 text-xs font-mono">
            <span className="text-neutral-400">Model:</span>
            <Badge variant="info">siglip-base-patch16-224 (768D)</Badge>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
          {/* Query Input Target (6 cols) */}
          <div className="lg:col-span-6 space-y-3">
            <label className="text-xs font-semibold text-neutral-300 uppercase tracking-wider flex items-center gap-2">
              <FileImage className="w-3.5 h-3.5 text-blue-400" />
              Query Source Modality
            </label>

            {queryMode === "upload" ? (
              <div
                className="border-2 border-dashed border-white/10 hover:border-blue-500/40 rounded-xl p-5 text-center cursor-pointer bg-neutral-950/50 transition-all min-h-[140px] flex flex-col items-center justify-center"
                onClick={() => document.getElementById("search-query-file")?.click()}
              >
                <input
                  id="search-query-file"
                  type="file"
                  accept="image/jpeg,image/png,image/webp"
                  className="hidden"
                  onChange={handleQueryFileChange}
                />
                {queryPreview ? (
                  <div className="flex flex-col items-center space-y-2">
                    <img src={queryPreview} alt="Query Preview" className="max-h-28 rounded-lg object-contain border border-white/10 shadow" />
                    <span className="text-[11px] text-blue-400">Click to change query image</span>
                  </div>
                ) : (
                  <div className="space-y-1.5">
                    <div className="w-9 h-9 rounded-full bg-blue-500/10 text-blue-400 mx-auto flex items-center justify-center">
                      <Upload className="w-4 h-4" />
                    </div>
                    <div className="text-xs font-medium text-neutral-200">Drop query image here or click to browse</div>
                    <div className="text-[11px] text-neutral-500">Supports JPEG, PNG, WebP up to 20MB</div>
                  </div>
                )}
              </div>
            ) : (
              <div className="space-y-2 bg-neutral-950 p-4 rounded-xl border border-white/10 min-h-[140px] flex flex-col justify-center">
                <label className="text-xs text-neutral-400">Select Indexed Memory Record</label>
                {memoryRecords.length === 0 ? (
                  <div className="text-xs text-amber-400 flex items-center gap-2">
                    <AlertCircle className="w-4 h-4" />
                    No indexed items in Visual Memory yet. Please index an image first.
                  </div>
                ) : (
                  <select
                    value={selectedRecordId}
                    onChange={(e) => setSelectedRecordId(e.target.value)}
                    className="w-full bg-neutral-900 border border-white/10 rounded-lg px-3 py-2 text-xs text-neutral-200 focus:outline-none focus:border-blue-500"
                  >
                    {memoryRecords.map((rec) => (
                      <option key={rec.id} value={rec.id}>
                        {rec.id} — ({rec.image_metadata?.width || 224}x{rec.image_metadata?.height || 224}) [{rec.tags?.join(", ") || "no tags"}]
                      </option>
                    ))}
                  </select>
                )}
              </div>
            )}
          </div>

          {/* Search Controls (6 cols) */}
          <div className="lg:col-span-6 space-y-4">
            <label className="text-xs font-semibold text-neutral-300 uppercase tracking-wider flex items-center gap-2">
              <Sliders className="w-3.5 h-3.5 text-purple-400" />
              Similarity Search Parameters
            </label>

            <div className="grid grid-cols-2 gap-4 text-xs">
              <div className="space-y-1.5 bg-neutral-950 p-3 rounded-lg border border-white/5">
                <div className="flex justify-between text-neutral-400">
                  <span>Top-K Retrieval</span>
                  <span className="font-mono text-purple-400 font-bold">{topK}</span>
                </div>
                <input
                  type="range"
                  min={1}
                  max={50}
                  value={topK}
                  onChange={(e) => setTopK(parseInt(e.target.value))}
                  className="w-full accent-purple-500"
                />
              </div>

              <div className="space-y-1.5 bg-neutral-950 p-3 rounded-lg border border-white/5">
                <div className="flex justify-between text-neutral-400">
                  <span>Minimum Threshold</span>
                  <span className="font-mono text-emerald-400 font-bold">{(threshold * 100).toFixed(0)}%</span>
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

            <div className="flex items-center justify-between bg-neutral-950 p-3 rounded-lg border border-white/5 text-xs">
              <span className="text-neutral-400 font-medium">Distance Metric</span>
              <select
                value={metric}
                onChange={(e) => setMetric(e.target.value)}
                className="bg-neutral-900 border border-white/10 rounded px-3 py-1 text-xs text-neutral-200 focus:outline-none"
              >
                <option value="cosine">Cosine Similarity (Normalized)</option>
                <option value="dot_product">Dot Product Inner Space</option>
                <option value="euclidean">Euclidean Distance (L2)</option>
              </select>
            </div>

            <Button
              variant="primary"
              size="lg"
              className="w-full justify-center"
              icon={<Sparkles className={`w-4 h-4 ${searching ? "animate-spin" : ""}`} />}
              onClick={handleSearchExecute}
              disabled={searching || (queryMode === "upload" && !queryFile) || (queryMode === "record" && !selectedRecordId)}
            >
              {searching ? "Extracting Features & Searching..." : "Execute Visual Search"}
            </Button>
          </div>
        </div>
      </Card>

      {/* Research Information Telemetry Bar */}
      {searchResults && (
        <Card className="p-4 bg-neutral-900/50 space-y-3 border-blue-500/20">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 text-xs border-b border-white/5 pb-2">
            <div className="flex items-center space-x-3">
              <span className="text-neutral-400 font-mono">Transaction ID:</span>
              <span className="font-mono text-blue-400 font-semibold">{searchResults.search_id}</span>
            </div>

            <div className="flex items-center space-x-3 text-neutral-400 font-mono">
              <span>Timestamp: {new Date(searchResults.timestamp).toLocaleTimeString()}</span>
              <span>•</span>
              <span>Metric: <strong className="text-neutral-200 uppercase">{searchResults.metric_used}</strong></span>
            </div>
          </div>

          <div className="grid grid-cols-2 sm:grid-cols-5 gap-3 text-xs font-mono">
            <div className="p-2.5 bg-neutral-950 rounded-lg border border-white/5">
              <div className="text-neutral-500 text-[10px]">Candidates Evaluated</div>
              <div className="text-base text-neutral-100 font-bold">{searchResults.candidate_count}</div>
            </div>

            <div className="p-2.5 bg-neutral-950 rounded-lg border border-white/5">
              <div className="text-neutral-500 text-[10px]">Matches Returned</div>
              <div className="text-base text-emerald-400 font-bold">{searchResults.returned_count}</div>
            </div>

            <div className="p-2.5 bg-neutral-950 rounded-lg border border-white/5">
              <div className="text-neutral-500 text-[10px]">Embedding Time</div>
              <div className="text-base text-amber-400 font-bold">{searchResults.embedding_time_ms.toFixed(1)}ms</div>
            </div>

            <div className="p-2.5 bg-neutral-950 rounded-lg border border-white/5">
              <div className="text-neutral-500 text-[10px]">Search Duration</div>
              <div className="text-base text-purple-400 font-bold">{searchResults.search_time_ms.toFixed(1)}ms</div>
            </div>

            <div className="p-2.5 bg-neutral-950 rounded-lg border border-white/5">
              <div className="text-neutral-500 text-[10px]">Total Execution</div>
              <div className="text-base text-blue-400 font-bold">{searchResults.total_execution_time_ms.toFixed(1)}ms</div>
            </div>
          </div>
        </Card>
      )}

      {/* Query vs Results Main Workspace */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left: Query Inspector (3 cols) */}
        <div className="lg:col-span-3 space-y-4">
          <Card className="p-4 space-y-3">
            <div className="text-xs font-semibold text-neutral-300 uppercase tracking-wider flex items-center gap-2 border-b border-white/10 pb-2">
              <Zap className="w-3.5 h-3.5 text-amber-400" />
              Query Source
            </div>

            {queryMode === "upload" && queryPreview ? (
              <div className="space-y-2">
                <img src={queryPreview} alt="Query Source" className="w-full max-h-48 rounded-lg object-contain border border-blue-500/30 bg-neutral-950" />
                <div className="p-2.5 bg-neutral-950 rounded text-[11px] font-mono text-neutral-400 space-y-1 border border-white/5">
                  <div>Source: Uploaded File</div>
                  <div>Name: {queryFile?.name || "query_image"}</div>
                  <div>Size: {((queryFile?.size || 0) / 1024).toFixed(1)} KB</div>
                </div>
              </div>
            ) : queryMode === "record" && selectedRecordId ? (
              <div className="p-3 bg-neutral-950 rounded-lg border border-purple-500/30 space-y-2 text-xs font-mono">
                <div className="text-purple-400 font-bold">Record ID: {selectedRecordId}</div>
                <div className="text-neutral-400">Indexed Record in Visual Memory</div>
              </div>
            ) : (
              <div className="p-6 text-center text-xs text-neutral-500 border border-dashed border-white/10 rounded-lg">
                No query source selected
              </div>
            )}
          </Card>
        </div>

        {/* Right: Ranked Results Gallery (9 cols) */}
        <div className="lg:col-span-9 space-y-4">
          {searchResults ? (
            <Card className="p-5 space-y-4">
              <div className="flex items-center justify-between border-b border-white/10 pb-3">
                <h3 className="text-sm font-semibold text-neutral-200 flex items-center gap-2">
                  <BarChart3 className="w-4 h-4 text-emerald-400" />
                  Ranked Visual Similarity Results ({searchResults.results.length})
                </h3>
                <span className="text-xs text-neutral-400 font-mono">
                  Showing Top-{topK} (Threshold &gt;= {(threshold * 100).toFixed(0)}%)
                </span>
              </div>

              {searchResults.results.length === 0 ? (
                <div className="py-16 text-center text-xs text-neutral-400 space-y-2">
                  <div className="w-10 h-10 rounded-full bg-neutral-900 text-neutral-500 mx-auto flex items-center justify-center">
                    <Search className="w-5 h-5" />
                  </div>
                  <div>No visual memory matches satisfied minimum similarity threshold ({(threshold * 100).toFixed(0)}%).</div>
                  <div className="text-neutral-500">Try lowering threshold or adding more images to Visual Memory.</div>
                </div>
              ) : (
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {searchResults.results.map((item) => (
                    <div
                      key={item.id}
                      className="p-4 bg-neutral-950 border border-white/10 hover:border-blue-500/40 rounded-xl space-y-3 transition-all flex flex-col justify-between"
                    >
                      <div className="space-y-3">
                        <div className="flex items-center justify-between">
                          <div className="flex items-center space-x-2">
                            <span className="w-7 h-7 rounded-lg bg-blue-500/15 border border-blue-500/30 text-blue-400 font-mono font-bold text-xs flex items-center justify-center">
                              #{item.rank}
                            </span>
                            <span className="font-mono text-xs font-semibold text-neutral-200 truncate max-w-[140px]">
                              {item.id}
                            </span>
                          </div>

                          <div className="text-right">
                            <span className="text-base font-mono font-bold text-emerald-400">
                              {(item.similarity_score * 100).toFixed(1)}%
                            </span>
                            <div className="text-[10px] text-neutral-500 font-mono uppercase">Similarity</div>
                          </div>
                        </div>

                        <div className="grid grid-cols-2 gap-2 text-[11px] font-mono bg-neutral-900/60 p-2.5 rounded-lg border border-white/5">
                          <div className="text-neutral-400">Distance: <span className="text-neutral-200">{item.distance.toFixed(4)}</span></div>
                          <div className="text-neutral-400 truncate">Format: <span className="text-neutral-200">{item.image_metadata?.format || "RGB"}</span></div>
                          <div className="text-neutral-400 truncate">Resolution: <span className="text-neutral-200">{item.image_metadata?.width || 224}x{item.image_metadata?.height || 224}</span></div>
                          <div className="text-neutral-400 truncate">Model: <span className="text-neutral-200">siglip</span></div>
                        </div>

                        {item.tags.length > 0 && (
                          <div className="flex flex-wrap gap-1">
                            {item.tags.map((t) => (
                              <span key={t} className="text-[10px] bg-blue-500/10 text-blue-300 px-2 py-0.5 rounded border border-blue-500/20 flex items-center gap-1">
                                <Tag className="w-2.5 h-2.5" />
                                {t}
                              </span>
                            ))}
                          </div>
                        )}
                      </div>

                      <Button
                        variant="secondary"
                        size="sm"
                        className="w-full justify-center mt-2 text-xs"
                        icon={<Info className="w-3.5 h-3.5" />}
                        onClick={() => setInspectItem(item)}
                      >
                        Inspect Match Details
                      </Button>
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
                <h4 className="text-base font-semibold text-neutral-200">Visual Search Research Workspace</h4>
                <p className="text-xs text-neutral-400">
                  Select a query source on the top panel and click &quot;Execute Visual Search&quot; to compute feature matrix distance and rank candidates.
                </p>
              </div>
            </Card>
          )}
        </div>
      </div>

      {/* Match Inspection Modal */}
      {inspectItem && (
        <div className="fixed inset-0 z-50 bg-black/80 backdrop-blur-sm flex items-center justify-center p-4">
          <div className="bg-[#111] border border-white/10 rounded-2xl max-w-lg w-full p-6 space-y-5 shadow-2xl">
            <div className="flex items-center justify-between border-b border-white/10 pb-3">
              <div className="flex items-center space-x-2">
                <Badge variant="success">Rank #{inspectItem.rank}</Badge>
                <h3 className="text-sm font-semibold text-neutral-200 font-mono">{inspectItem.id}</h3>
              </div>
              <button onClick={() => setInspectItem(null)} className="text-neutral-400 hover:text-white transition-colors">
                <X className="w-5 h-5" />
              </button>
            </div>

            <div className="space-y-4 text-xs font-mono">
              <div className="grid grid-cols-2 gap-3">
                <div className="p-3 bg-neutral-950 rounded-lg border border-white/5 space-y-1">
                  <div className="text-neutral-500 uppercase text-[10px]">Similarity Score</div>
                  <div className="text-lg text-emerald-400 font-bold">{(inspectItem.similarity_score * 100).toFixed(2)}%</div>
                </div>

                <div className="p-3 bg-neutral-950 rounded-lg border border-white/5 space-y-1">
                  <div className="text-neutral-500 uppercase text-[10px]">Distance Metric</div>
                  <div className="text-lg text-purple-300 font-bold">{inspectItem.distance.toFixed(4)}</div>
                </div>
              </div>

              <div className="p-3 bg-neutral-950 rounded-lg border border-white/5 space-y-2">
                <div className="text-neutral-400 font-semibold border-b border-white/5 pb-1">Image Parameters</div>
                <div className="grid grid-cols-2 gap-2 text-neutral-300">
                  <div>Width: {inspectItem.image_metadata?.width || "N/A"} px</div>
                  <div>Height: {inspectItem.image_metadata?.height || "N/A"} px</div>
                  <div>Format: {inspectItem.image_metadata?.format || "RGB"}</div>
                  <div>Mode: {inspectItem.image_metadata?.mode || "RGB"}</div>
                </div>
              </div>

              <div className="p-3 bg-neutral-950 rounded-lg border border-white/5 space-y-2">
                <div className="text-neutral-400 font-semibold border-b border-white/5 pb-1">Embedding Metadata</div>
                <div className="space-y-1 text-neutral-300">
                  <div>Model: {inspectItem.embedding_model}</div>
                  <div>Dimension: 768-D Dense Vector</div>
                  <div>Indexed At: {new Date(inspectItem.indexed_at).toLocaleString()}</div>
                </div>
              </div>
            </div>

            <Button variant="secondary" size="sm" className="w-full justify-center" onClick={() => setInspectItem(null)}>
              Close Inspector
            </Button>
          </div>
        </div>
      )}
    </div>
  );
}
