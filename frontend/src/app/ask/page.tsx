"use client";

import React, { useEffect, useState } from "react";
import Link from "next/link";
import {
  Sparkles,
  Search,
  MessageSquare,
  History,
  ShieldCheck,
  RotateCcw,
  ExternalLink,
  ChevronRight,
  Filter,
  CheckCircle2,
  AlertCircle,
  Clock,
  Layers,
  Database,
  Video,
  BarChart2,
  Cpu,
  Eye,
  Info,
} from "lucide-react";
import { PageHeader } from "@/components/ui/PageHeader";
import { Button } from "@/components/ui/Button";
import { Badge } from "@/components/ui/Badge";

interface VisionEvidenceItem {
  evidence_id: string;
  evidence_type: string;
  title: string;
  description: string;
  thumbnail_uri?: string;
  sample_id?: string;
  dataset_id?: string;
  model_id?: string;
  video_id?: string;
  timestamp_sec?: number;
  frame_idx?: number;
  track_id?: number;
  event_id?: string;
  bbox?: number[];
  confidence?: number;
  class_name?: string;
  iou?: number;
  action_link: string;
  metadata?: Record<string, any>;
}

interface VisionQuery {
  query_id: string;
  user_query: string;
  query_type: string;
  target: Record<string, any>;
  filters: Record<string, any>;
  structured_query: Record<string, any>;
  execution_result: Record<string, any>;
  answer: string;
  evidence: VisionEvidenceItem[];
  status: string;
  clarification_needed?: string;
  clarification_options?: string[];
  grounding_verified: boolean;
  reproducibility_hash: string;
  execution_time_ms: number;
  created_timestamp: string;
}

interface HistoryItem {
  query_id: string;
  user_query: string;
  query_type: string;
  status: string;
  results_count: number;
  created_timestamp: string;
  execution_time_ms: number;
}

interface SuggestedPrompt {
  category: string;
  icon: React.ReactNode;
  prompts: string[];
}

export default function AskVisionForgePage() {
  const [inputText, setInputText] = useState("");
  const [loading, setLoading] = useState(false);
  const [currentQuery, setCurrentQuery] = useState<VisionQuery | null>(null);
  const [conversation, setConversation] = useState<VisionQuery[]>([]);
  const [history, setHistory] = useState<HistoryItem[]>([]);
  const [selectedDomain, setSelectedDomain] = useState<string>("all");

  const promptCategories: SuggestedPrompt[] = [
    {
      category: "Failure Analysis",
      icon: <AlertCircle className="w-3.5 h-3.5 text-rose-400" />,
      prompts: [
        "Show helmet failures with confidence below 0.50",
        "Why is sample 1024 considered a failure?",
        "Show false positives for person class",
      ],
    },
    {
      category: "Video & Events",
      icon: <Video className="w-3.5 h-3.5 text-cyan-400" />,
      prompts: [
        "Which objects entered Zone A?",
        "Which person stayed longer than 3 seconds?",
        "Show trajectory and velocity for Track 1",
      ],
    },
    {
      category: "Dataset Intelligence",
      icon: <Database className="w-3.5 h-3.5 text-emerald-400" />,
      prompts: [
        "How many samples are in the dataset?",
        "Show underrepresented classes in safety_v2",
        "Show duplicate candidates and quality issues",
      ],
    },
    {
      category: "Model Comparison",
      icon: <BarChart2 className="w-3.5 h-3.5 text-purple-400" />,
      prompts: [
        "Compare model yolo11s.pt and yolo11m.pt",
        "Which model performs best on the benchmark suite?",
      ],
    },
  ];

  // Fetch initial history
  useEffect(() => {
    fetchHistory();
  }, []);

  const fetchHistory = async () => {
    try {
      const res = await fetch("/api/v1/multimodal/history");
      if (res.ok) {
        const data = await res.json();
        setHistory(data);
      }
    } catch (e) {
      console.error("Failed to load query history:", e);
    }
  };

  const handleAsk = async (queryText?: string) => {
    const textToSend = queryText || inputText;
    if (!textToSend.trim()) return;

    setLoading(true);
    try {
      const prev = currentQuery
        ? {
            query_type: currentQuery.query_type,
            filters: currentQuery.filters,
            target: currentQuery.target,
          }
        : null;

      const res = await fetch("/api/v1/multimodal/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query: textToSend,
          context: {
            session_id: "ask_session_1",
            previous_query: prev,
          },
        }),
      });

      if (res.ok) {
        const data: VisionQuery = await res.json();
        setCurrentQuery(data);
        setConversation((prev) => [...prev, data]);
        setInputText("");
        fetchHistory();
      }
    } catch (e) {
      console.error("Query request failed:", e);
    } finally {
      setLoading(false);
    }
  };

  const handleReplay = async (queryId: string) => {
    setLoading(true);
    try {
      const res = await fetch(`/api/v1/multimodal/queries/${queryId}/replay`, {
        method: "POST",
      });
      if (res.ok) {
        const data: VisionQuery = await res.json();
        setCurrentQuery(data);
        setConversation((prev) => [...prev, data]);
      }
    } catch (e) {
      console.error("Replay failed:", e);
    } finally {
      setLoading(false);
    }
  };

  const getEvidenceIcon = (type: string) => {
    switch (type) {
      case "FAILURE_SAMPLE":
        return <AlertCircle className="w-4 h-4 text-rose-400" />;
      case "TEMPORAL_EVENT":
        return <Clock className="w-4 h-4 text-cyan-400" />;
      case "DATASET_PROFILE":
        return <Database className="w-4 h-4 text-emerald-400" />;
      case "MODEL_EVALUATION":
        return <BarChart2 className="w-4 h-4 text-purple-400" />;
      case "SIMILAR_SAMPLE":
        return <Eye className="w-4 h-4 text-amber-400" />;
      default:
        return <Layers className="w-4 h-4 text-blue-400" />;
    }
  };

  return (
    <div className="min-h-screen bg-[#070709] text-neutral-200 font-sans pb-16">
      {/* Workbench Header */}
      <div className="border-b border-white/10 bg-[#0d0d12]/90 backdrop-blur-md px-6 py-4 sticky top-14 z-20">
        <div className="max-w-7xl mx-auto flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-cyan-600/30 to-blue-600/30 border border-cyan-500/40 flex items-center justify-center text-cyan-300 shadow-lg shadow-cyan-950/40">
              <Sparkles className="w-5 h-5" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h1 className="text-lg font-semibold text-white tracking-tight">Ask VisionForge</h1>
                <Badge variant="info" size="sm" className="font-mono text-[10px]">
                  MULTIMODAL VLM
                </Badge>
                <span className="flex items-center gap-1 text-[11px] font-mono text-emerald-400 bg-emerald-950/40 border border-emerald-800/40 px-2 py-0.5 rounded-full">
                  <ShieldCheck className="w-3 h-3" />
                  GROUNDED FACTUAL SYNTHESIS
                </span>
              </div>
              <p className="text-xs text-neutral-400 mt-0.5">
                Natural-language visual intelligence strictly verified against observable vision data
              </p>
            </div>
          </div>

          <div className="flex items-center gap-2 text-xs text-neutral-400 font-mono">
            <span>Deterministic Grounding Validator:</span>
            <span className="text-emerald-400 font-semibold">ACTIVE</span>
          </div>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-6 pt-6 grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Main Conversation & Question Area (8 cols) */}
        <div className="lg:col-span-8 space-y-6">
          {/* Query Input Box */}
          <div className="rounded-2xl bg-neutral-900/80 border border-white/15 p-4 shadow-2xl backdrop-blur-md focus-within:border-cyan-500/60 transition-all">
            <div className="flex items-center gap-3">
              <Search className="w-5 h-5 text-neutral-400" />
              <input
                type="text"
                value={inputText}
                onChange={(e) => setInputText(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && !loading && handleAsk()}
                placeholder="Ask about image detections, video events, failure gallery, dataset health, or model metrics..."
                className="w-full bg-transparent border-none text-white text-sm placeholder-neutral-500 focus:outline-none"
                disabled={loading}
              />
              <Button
                variant="primary"
                size="sm"
                onClick={() => handleAsk()}
                disabled={loading || !inputText.trim()}
                className="shrink-0 bg-cyan-600 hover:bg-cyan-500 text-black font-semibold shadow-md shadow-cyan-900/30"
              >
                {loading ? "Verifying..." : "Ask"}
              </Button>
            </div>

            {/* Suggested Prompt Chips */}
            <div className="mt-4 pt-3 border-t border-white/10 space-y-2">
              <div className="text-[11px] font-mono text-neutral-400 flex items-center justify-between">
                <span>SUGGESTED RESEARCH QUESTIONS:</span>
              </div>
              <div className="flex flex-wrap gap-2">
                {promptCategories.flatMap((cat) =>
                  cat.prompts.map((p) => (
                    <button
                      key={p}
                      onClick={() => handleAsk(p)}
                      className="px-2.5 py-1 rounded-lg bg-neutral-800/80 hover:bg-neutral-800 border border-white/10 hover:border-cyan-500/40 text-neutral-300 hover:text-cyan-300 text-xs transition-all flex items-center gap-1.5 cursor-pointer text-left"
                    >
                      {cat.icon}
                      <span>{p}</span>
                    </button>
                  ))
                )}
              </div>
            </div>
          </div>

          {/* Active Query Output */}
          {currentQuery ? (
            <div className="space-y-6">
              {/* Answer Box */}
              <div className="rounded-2xl bg-neutral-900/90 border border-white/15 p-6 shadow-xl space-y-4">
                <div className="flex items-center justify-between border-b border-white/10 pb-3">
                  <div className="flex items-center gap-2">
                    <span className="text-xs font-mono px-2 py-0.5 rounded bg-neutral-800 border border-white/10 text-cyan-300">
                      {currentQuery.query_type}
                    </span>
                    <span className="text-xs text-neutral-400 font-mono">
                      Query ID: <strong className="text-white">{currentQuery.query_id}</strong>
                    </span>
                  </div>

                  <div className="flex items-center gap-2 font-mono text-[11px]">
                    {currentQuery.grounding_verified ? (
                      <span className="flex items-center gap-1 text-emerald-400 bg-emerald-950/40 border border-emerald-800/30 px-2 py-0.5 rounded">
                        <CheckCircle2 className="w-3 h-3" /> GROUNDING VERIFIED
                      </span>
                    ) : (
                      <span className="flex items-center gap-1 text-amber-400 bg-amber-950/40 border border-amber-800/30 px-2 py-0.5 rounded">
                        <AlertCircle className="w-3 h-3" /> RE-GROUNDED FALLBACK
                      </span>
                    )}
                    <span className="text-neutral-500">|</span>
                    <span className="text-neutral-400">{currentQuery.execution_time_ms} ms</span>
                  </div>
                </div>

                {/* User Prompt Echo */}
                <div className="text-sm font-medium text-neutral-300 italic border-l-2 border-cyan-500 pl-3">
                  "{currentQuery.user_query}"
                </div>

                {/* Clarification Prompt if Ambiguous */}
                {currentQuery.status === "AMBIGUOUS" && currentQuery.clarification_needed && (
                  <div className="p-4 rounded-xl bg-amber-950/20 border border-amber-500/30 space-y-3">
                    <div className="flex items-center gap-2 text-amber-400 font-semibold text-xs">
                      <AlertCircle className="w-4 h-4" />
                      <span>CLARIFICATION REQUIRED</span>
                    </div>
                    <p className="text-xs text-neutral-300">{currentQuery.clarification_needed}</p>
                    <div className="flex flex-wrap gap-2 pt-1">
                      {currentQuery.clarification_options?.map((opt) => (
                        <button
                          key={opt}
                          onClick={() => handleAsk(`${currentQuery.user_query} for ${opt}`)}
                          className="px-3 py-1 rounded bg-amber-900/40 hover:bg-amber-900/60 border border-amber-500/40 text-amber-200 text-xs font-mono transition-colors"
                        >
                          {opt}
                        </button>
                      ))}
                    </div>
                  </div>
                )}

                {/* Grounded Natural Language Answer */}
                <div className="p-4 rounded-xl bg-neutral-950 border border-white/10 space-y-2">
                  <div className="text-xs font-mono text-neutral-400 flex items-center gap-1">
                    <Sparkles className="w-3.5 h-3.5 text-cyan-400" />
                    <span>GROUNDED VISUAL ANSWER</span>
                  </div>
                  <p className="text-sm text-neutral-100 font-sans leading-relaxed">
                    {currentQuery.answer}
                  </p>
                </div>

                {/* Structured Evidence References */}
                <div className="space-y-3 pt-2">
                  <div className="flex items-center justify-between">
                    <h3 className="text-xs font-mono text-neutral-400 uppercase tracking-wider flex items-center gap-1.5">
                      <Layers className="w-3.5 h-3.5 text-cyan-400" />
                      <span>Structured Evidence Records ({currentQuery.evidence.length})</span>
                    </h3>
                    <span className="text-[11px] font-mono text-neutral-500">
                      Click evidence card to navigate directly
                    </span>
                  </div>

                  {currentQuery.evidence.length === 0 ? (
                    <div className="p-4 rounded-xl bg-neutral-950/60 border border-dashed border-white/10 text-center text-xs text-neutral-500">
                      No explicit visual evidence attachments found for this query result.
                    </div>
                  ) : (
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                      {currentQuery.evidence.map((evi) => (
                        <Link
                          key={evi.evidence_id}
                          href={evi.action_link}
                          className="group p-3 rounded-xl bg-neutral-950/70 border border-white/10 hover:border-cyan-500/40 hover:bg-neutral-900/80 transition-all flex flex-col justify-between space-y-2.5"
                        >
                          <div className="space-y-1.5">
                            <div className="flex items-center justify-between">
                              <span className="flex items-center gap-1.5 text-xs font-semibold text-white group-hover:text-cyan-300 transition-colors">
                                {getEvidenceIcon(evi.evidence_type)}
                                <span>{evi.title}</span>
                              </span>
                              <ExternalLink className="w-3.5 h-3.5 text-neutral-500 group-hover:text-cyan-400 transition-colors" />
                            </div>
                            <p className="text-xs text-neutral-400 line-clamp-2 leading-relaxed">
                              {evi.description}
                            </p>
                          </div>

                          {/* Evidence Meta Badges */}
                          <div className="flex flex-wrap items-center gap-1.5 pt-1 font-mono text-[10px]">
                            {evi.class_name && (
                              <span className="px-1.5 py-0.5 rounded bg-neutral-800 text-neutral-300 border border-white/10">
                                class: {evi.class_name}
                              </span>
                            )}
                            {evi.confidence !== undefined && (
                              <span className="px-1.5 py-0.5 rounded bg-cyan-950/40 text-cyan-300 border border-cyan-800/30">
                                conf: {(evi.confidence * 100).toFixed(0)}%
                              </span>
                            )}
                            {evi.iou !== undefined && (
                              <span className="px-1.5 py-0.5 rounded bg-rose-950/40 text-rose-300 border border-rose-800/30">
                                IoU: {evi.iou.toFixed(2)}
                              </span>
                            )}
                            {evi.timestamp_sec !== undefined && (
                              <span className="px-1.5 py-0.5 rounded bg-neutral-800 text-neutral-300 border border-white/10">
                                t={evi.timestamp_sec.toFixed(1)}s
                              </span>
                            )}
                          </div>
                        </Link>
                      ))}
                    </div>
                  )}
                </div>
              </div>
            </div>
          ) : (
            <div className="rounded-2xl bg-neutral-900/40 border border-dashed border-white/10 p-12 text-center space-y-4">
              <div className="w-12 h-12 rounded-2xl bg-neutral-800/60 border border-white/10 flex items-center justify-center text-neutral-400 mx-auto">
                <Sparkles className="w-6 h-6 text-cyan-400" />
              </div>
              <div className="space-y-1 max-w-md mx-auto">
                <h3 className="text-sm font-semibold text-white">Ask anything across your Vision Pipeline</h3>
                <p className="text-xs text-neutral-400">
                  Type a natural language question or select one of the suggested prompts above. All answers
                  are factually grounded in verifiable detections, trajectories, events, and evaluations.
                </p>
              </div>
            </div>
          )}
        </div>

        {/* Right Sidebar: Query Provenance & History (4 cols) */}
        <div className="lg:col-span-4 space-y-6">
          {/* Query Provenance Card */}
          {currentQuery && (
            <div className="rounded-2xl bg-neutral-900/80 border border-white/15 p-5 space-y-4 shadow-xl">
              <div className="flex items-center gap-2 border-b border-white/10 pb-3">
                <Info className="w-4 h-4 text-cyan-400" />
                <h3 className="text-xs font-semibold text-white tracking-tight uppercase font-mono">
                  Query Provenance & DSL
                </h3>
              </div>

              <div className="space-y-3 text-xs">
                <div>
                  <span className="text-neutral-500 font-mono text-[10px] block mb-1">RESOLVED TARGETS:</span>
                  <div className="p-2 rounded bg-neutral-950 border border-white/10 font-mono text-[11px] text-neutral-300">
                    {Object.keys(currentQuery.target).length > 0 ? (
                      JSON.stringify(currentQuery.target, null, 2)
                    ) : (
                      <span className="text-neutral-500">None</span>
                    )}
                  </div>
                </div>

                <div>
                  <span className="text-neutral-500 font-mono text-[10px] block mb-1">ACTIVE FILTERS:</span>
                  <div className="p-2 rounded bg-neutral-950 border border-white/10 font-mono text-[11px] text-neutral-300">
                    {Object.keys(currentQuery.filters).length > 0 ? (
                      JSON.stringify(currentQuery.filters, null, 2)
                    ) : (
                      <span className="text-neutral-500">None</span>
                    )}
                  </div>
                </div>

                <div>
                  <span className="text-neutral-500 font-mono text-[10px] block mb-1">
                    REPRODUCIBILITY HASH:
                  </span>
                  <div className="p-2 rounded bg-neutral-950 border border-white/10 font-mono text-[11px] text-cyan-300 break-all">
                    {currentQuery.reproducibility_hash}
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Query History */}
          <div className="rounded-2xl bg-neutral-900/80 border border-white/15 p-5 space-y-4 shadow-xl">
            <div className="flex items-center justify-between border-b border-white/10 pb-3">
              <div className="flex items-center gap-2">
                <History className="w-4 h-4 text-cyan-400" />
                <h3 className="text-xs font-semibold text-white tracking-tight uppercase font-mono">
                  Query History ({history.length})
                </h3>
              </div>
              <Button
                variant="ghost"
                size="sm"
                onClick={fetchHistory}
                className="text-[10px] font-mono h-6 px-2 text-neutral-400 hover:text-white"
              >
                Refresh
              </Button>
            </div>

            {history.length === 0 ? (
              <div className="text-center py-6 text-xs text-neutral-500">No previous queries recorded.</div>
            ) : (
              <div className="space-y-2 max-h-96 overflow-y-auto pr-1">
                {history.map((item) => (
                  <div
                    key={item.query_id}
                    className="p-2.5 rounded-xl bg-neutral-950/70 border border-white/10 hover:border-cyan-500/30 transition-all space-y-1.5"
                  >
                    <div className="flex items-center justify-between">
                      <span className="text-[10px] font-mono px-1.5 py-0.2 rounded bg-neutral-800 text-cyan-300 border border-white/10">
                        {item.query_type}
                      </span>
                      <span className="text-[10px] font-mono text-neutral-500">
                        {item.results_count} results
                      </span>
                    </div>

                    <p className="text-xs text-neutral-200 line-clamp-1 font-medium">"{item.user_query}"</p>

                    <div className="flex items-center justify-between pt-1 text-[10px] font-mono text-neutral-500">
                      <span>{new Date(item.created_timestamp).toLocaleTimeString()}</span>
                      <button
                        onClick={() => handleReplay(item.query_id)}
                        className="text-cyan-400 hover:text-cyan-300 flex items-center gap-1 font-sans cursor-pointer"
                      >
                        <RotateCcw className="w-3 h-3" /> Replay
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
