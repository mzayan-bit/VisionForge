"use client";

import React, { useEffect, useState, useRef } from "react";
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
  AlertCircle,
  Clock,
  Layers,
  Database,
  Video,
  BarChart2,
  Cpu,
  Eye,
  Info,
  CheckCircle2,
  RefreshCw,
  Send,
  Trash2,
  HelpCircle,
  Hash,
  Activity,
  Maximize2,
  Compass,
} from "lucide-react";
import { PageHeader } from "@/components/ui/PageHeader";
import { Button } from "@/components/ui/Button";
import { Badge } from "@/components/ui/Badge";
import { Card } from "@/components/ui/Card";

export interface VisionEvidenceItem {
  evidence_id: string;
  evidence_type: string;
  title: string;
  description: string;
  thumbnail_uri?: string | null;
  sample_id?: string | null;
  dataset_id?: string | null;
  model_id?: string | null;
  video_id?: string | null;
  timestamp_sec?: number | null;
  frame_idx?: number | null;
  track_id?: number | null;
  event_id?: string | null;
  bbox?: number[] | null;
  confidence?: number | null;
  class_name?: string | null;
  iou?: number | null;
  action_link: string;
  metadata?: Record<string, any>;
}

export interface VisionQuery {
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
  clarification_needed?: string | null;
  clarification_options?: string[] | null;
  grounding_verified: boolean;
  reproducibility_hash: string;
  execution_time_ms: number;
  created_timestamp: string;
}

export interface HistoryItem {
  query_id: string;
  user_query: string;
  query_type: string;
  status: string;
  results_count: number;
  created_timestamp: string;
  execution_time_ms: number;
}

interface MessageItem {
  id: string;
  role: "user" | "assistant";
  text: string;
  query?: VisionQuery;
  error?: string;
  timestamp: string;
}

export default function AskVisionForgePage() {
  const [inputText, setInputText] = useState("");
  const [loading, setLoading] = useState(false);
  const [backendStatus, setBackendStatus] = useState<"checking" | "connected" | "offline">("checking");
  const [conversation, setConversation] = useState<MessageItem[]>([]);
  const [history, setHistory] = useState<HistoryItem[]>([]);
  const [historyLoading, setHistoryLoading] = useState(false);
  const [showProvenanceFor, setShowProvenanceFor] = useState<string | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [isMounted, setIsMounted] = useState(false);

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const promptCategories = [
    {
      id: "video",
      category: "Video & Spatiotemporal",
      icon: Video,
      color: "text-cyan-400 border-cyan-500/30 bg-cyan-950/20 hover:border-cyan-500/60",
      prompts: [
        "Which objects entered Zone A?",
        "Who stayed longer than 3 seconds in the zone?",
        "What happened at 12.4 seconds in the video?",
        "Show trajectory and velocity for Track 1",
      ],
    },
    {
      id: "models",
      category: "Models & Evaluation",
      icon: BarChart2,
      color: "text-purple-400 border-purple-500/30 bg-purple-950/20 hover:border-purple-500/60",
      prompts: [
        "Which model performs best on the benchmark suite?",
        "What was the mAP@50 of the latest model?",
        "Compare model yolo11s.pt and yolo11m.pt",
      ],
    },
    {
      id: "failures",
      category: "Failure Analysis",
      icon: AlertCircle,
      color: "text-rose-400 border-rose-500/30 bg-rose-950/20 hover:border-rose-500/60",
      prompts: [
        "Show helmet failures with confidence below 0.50",
        "What are the most common false positives?",
        "Why is sample 1024 considered a failure?",
      ],
    },
    {
      id: "datasets",
      category: "Dataset Intelligence",
      icon: Database,
      color: "text-emerald-400 border-emerald-500/30 bg-emerald-950/20 hover:border-emerald-500/60",
      prompts: [
        "How many samples are in the dataset?",
        "Show underrepresented classes in safety_v2",
        "Show duplicate candidates and quality issues",
      ],
    },
  ];

  // Initial load
  useEffect(() => {
    setIsMounted(true);
    checkHealth();
    fetchHistory();
  }, []);

  // Scroll to bottom of conversation
  useEffect(() => {
    if (conversation.length > 0) {
      messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    }
  }, [conversation, loading]);

  const checkHealth = async () => {
    try {
      const res = await fetch("/api/v1/health");
      if (res.ok) {
        setBackendStatus("connected");
      } else {
        setBackendStatus("offline");
      }
    } catch {
      setBackendStatus("offline");
    }
  };

  const fetchHistory = async () => {
    setHistoryLoading(true);
    try {
      const res = await fetch("/api/v1/multimodal/history");
      if (res.ok) {
        const data = await res.json();
        if (Array.isArray(data)) {
          setHistory(data);
        } else {
          setHistory([]);
        }
      } else {
        setHistory([]);
      }
    } catch (e) {
      console.warn("Failed to load query history:", e);
      setHistory([]);
    } finally {
      setHistoryLoading(false);
    }
  };

  const handleAsk = async (queryText?: string) => {
    const textToSend = (queryText || inputText).trim();
    if (!textToSend || loading) return;

    setErrorMessage(null);
    const userMsgId = `user_${Date.now()}`;
    const assistantMsgId = `asst_${Date.now()}`;
    const nowIso = new Date().toISOString();

    // Append user message immediately
    const userMsg: MessageItem = {
      id: userMsgId,
      role: "user",
      text: textToSend,
      timestamp: nowIso,
    };
    setConversation((prev) => [...prev, userMsg]);
    setInputText("");
    setLoading(true);

    try {
      const prevQuery = conversation
        .slice()
        .reverse()
        .find((m) => m.query)?.query;

      const prev = prevQuery
        ? {
            query_type: prevQuery.query_type,
            filters: prevQuery.filters,
            target: prevQuery.target,
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
        setBackendStatus("connected");
        const asstMsg: MessageItem = {
          id: assistantMsgId,
          role: "assistant",
          text: data.answer,
          query: data,
          timestamp: new Date().toISOString(),
        };
        setConversation((prev) => [...prev, asstMsg]);
        fetchHistory();
      } else {
        let errDetail = "VisionForge couldn't complete this query.";
        try {
          const errJson = await res.json();
          if (errJson.detail) errDetail = errJson.detail;
        } catch {
          // ignore parse error
        }
        setErrorMessage(errDetail);
        const asstErr: MessageItem = {
          id: assistantMsgId,
          role: "assistant",
          text: errDetail,
          error: errDetail,
          timestamp: new Date().toISOString(),
        };
        setConversation((prev) => [...prev, asstErr]);
      }
    } catch (e: any) {
      setBackendStatus("offline");
      const netErr = "VisionForge backend is currently unavailable. Please verify the server is running.";
      setErrorMessage(netErr);
      const asstErr: MessageItem = {
        id: assistantMsgId,
        role: "assistant",
        text: netErr,
        error: netErr,
        timestamp: new Date().toISOString(),
      };
      setConversation((prev) => [...prev, asstErr]);
    } finally {
      setLoading(false);
    }
  };

  const handleReplay = async (queryId: string) => {
    if (loading) return;
    setLoading(true);
    setErrorMessage(null);
    try {
      const res = await fetch(`/api/v1/multimodal/queries/${queryId}/replay`, {
        method: "POST",
      });
      if (res.ok) {
        const data: VisionQuery = await res.json();
        setBackendStatus("connected");
        const nowIso = new Date().toISOString();
        setConversation((prev) => [
          ...prev,
          {
            id: `replay_user_${Date.now()}`,
            role: "user",
            text: data.user_query,
            timestamp: nowIso,
          },
          {
            id: `replay_asst_${Date.now()}`,
            role: "assistant",
            text: data.answer,
            query: data,
            timestamp: nowIso,
          },
        ]);
      } else {
        setErrorMessage(`Failed to replay query '${queryId}'`);
      }
    } catch (e) {
      setErrorMessage("Could not connect to backend to replay query.");
    } finally {
      setLoading(false);
    }
  };

  const clearConversation = () => {
    setConversation([]);
    setErrorMessage(null);
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
    <div className="space-y-6">
      {/* Page Header */}
      <PageHeader
        title="Ask VisionForge"
        description="Ask natural-language questions about your CV datasets, models, experiments, and video analyses."
        breadcrumbs={["VisionForge", "Ask Assistant"]}
        actions={
          <div className="flex items-center gap-2">
            <div
              className={`flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-mono border ${
                backendStatus === "connected"
                  ? "bg-emerald-950/40 text-emerald-300 border-emerald-800/40"
                  : backendStatus === "offline"
                  ? "bg-rose-950/40 text-rose-300 border-rose-800/40"
                  : "bg-neutral-900 text-neutral-400 border-neutral-700"
              }`}
            >
              <div
                className={`w-2 h-2 rounded-full ${
                  backendStatus === "connected"
                    ? "bg-emerald-400 animate-pulse"
                    : backendStatus === "offline"
                    ? "bg-rose-400"
                    : "bg-neutral-500"
                }`}
              />
              <span>{backendStatus === "connected" ? "Backend Connected" : backendStatus === "offline" ? "Backend Offline" : "Checking..."}</span>
            </div>

            <div className="hidden sm:flex items-center gap-1 px-2.5 py-1 rounded-full text-xs font-mono bg-cyan-950/40 text-cyan-300 border border-cyan-800/40">
              <ShieldCheck className="w-3.5 h-3.5" />
              <span>Grounded Factual Synthesis</span>
            </div>

            {conversation.length > 0 && (
              <Button
                variant="secondary"
                size="sm"
                onClick={clearConversation}
                icon={<Trash2 className="w-3.5 h-3.5" />}
                className="text-xs"
              >
                Clear Thread
              </Button>
            )}
          </div>
        }
      />

      {/* Main Grid: 8 cols chat + 4 cols sidebar */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
        {/* Main Conversation Column */}
        <div className="lg:col-span-8 space-y-4">
          {/* Conversation Stream Container */}
          <div className="min-h-[420px] max-h-[680px] overflow-y-auto rounded-2xl bg-neutral-950/80 border border-white/10 p-4 md:p-6 flex flex-col justify-between custom-scrollbar space-y-6">
            {conversation.length === 0 ? (
              /* Empty State Banner & Suggestion Chips */
              <div className="my-auto py-8 text-center space-y-6">
                <div className="w-14 h-14 rounded-2xl bg-gradient-to-tr from-cyan-600/30 to-blue-600/30 border border-cyan-500/40 flex items-center justify-center text-cyan-300 mx-auto shadow-xl shadow-cyan-950/40">
                  <Sparkles className="w-7 h-7" />
                </div>

                <div className="space-y-1.5 max-w-lg mx-auto">
                  <h3 className="text-base font-semibold text-white">Ask VisionForge about your CV evidence</h3>
                  <p className="text-xs text-neutral-400 leading-relaxed">
                    Natural-language visual intelligence strictly verified against observable vision data: detections, trajectories, events, error taxonomy, and model metrics.
                  </p>
                </div>

                {/* Prompt Categories Grid */}
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 max-w-2xl mx-auto pt-2 text-left">
                  {promptCategories.map((cat) => {
                    const IconComponent = cat.icon;
                    return (
                      <div
                        key={cat.id}
                        className={`p-3.5 rounded-xl border transition-all ${cat.color}`}
                      >
                        <div className="flex items-center gap-2 mb-2 font-semibold text-xs text-neutral-200">
                          <IconComponent className="w-4 h-4 shrink-0" />
                          <span>{cat.category}</span>
                        </div>
                        <div className="space-y-1.5">
                          {cat.prompts.slice(0, 2).map((p) => (
                            <button
                              key={p}
                              onClick={() => handleAsk(p)}
                              disabled={loading}
                              className="w-full text-left text-xs text-neutral-400 hover:text-white transition-colors block py-0.5 truncate group"
                            >
                              <span className="text-neutral-600 group-hover:text-cyan-400 mr-1.5">→</span>
                              <span>{p}</span>
                            </button>
                          ))}
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            ) : (
              /* Chronological Message Thread */
              <div className="space-y-6">
                {conversation.map((msg) => (
                  <div key={msg.id} className="space-y-3">
                    {msg.role === "user" ? (
                      /* User Bubble */
                      <div className="flex items-start justify-end gap-2.5">
                        <div className="max-w-[85%] rounded-2xl rounded-tr-sm bg-cyan-600/20 border border-cyan-500/40 p-4 text-sm text-neutral-100 shadow-md">
                          <div className="text-xs font-mono text-cyan-300 mb-1 flex items-center justify-between gap-4">
                            <span>YOU</span>
                            {isMounted && (
                              <span className="text-[10px] text-neutral-400">
                                {new Date(msg.timestamp).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}
                              </span>
                            )}
                          </div>
                          <p className="font-medium">{msg.text}</p>
                        </div>
                        <div className="w-8 h-8 rounded-xl bg-cyan-900/40 border border-cyan-500/30 flex items-center justify-center text-cyan-300 shrink-0 text-xs font-bold">
                          U
                        </div>
                      </div>
                    ) : (
                      /* Assistant Bubble */
                      <div className="flex items-start gap-2.5">
                        <div className="w-8 h-8 rounded-xl bg-gradient-to-tr from-cyan-600/30 to-blue-600/30 border border-cyan-500/40 flex items-center justify-center text-cyan-300 shrink-0 shadow-md">
                          <Sparkles className="w-4 h-4" />
                        </div>

                        <div className="flex-1 max-w-[90%] rounded-2xl rounded-tl-sm bg-neutral-900/90 border border-white/10 p-5 space-y-4 shadow-xl">
                          {/* Query Header & Grounding Badge */}
                          <div className="flex flex-wrap items-center justify-between gap-2 border-b border-white/10 pb-3 font-mono text-xs">
                            <div className="flex items-center gap-2">
                              {msg.query ? (
                                <span className="px-2 py-0.5 rounded bg-neutral-800 text-cyan-300 border border-white/10">
                                  {msg.query.query_type}
                                </span>
                              ) : (
                                <span className="px-2 py-0.5 rounded bg-neutral-800 text-neutral-400 border border-white/10">
                                  ASSISTANT
                                </span>
                              )}
                              {msg.query && (
                                <span className="text-[11px] text-neutral-400">
                                  ID: <strong className="text-white">{msg.query.query_id}</strong>
                                </span>
                              )}
                            </div>

                            {msg.query && (
                              <div className="flex items-center gap-2 text-[11px]">
                                {msg.query.grounding_verified ? (
                                  <span className="flex items-center gap-1 text-emerald-400 bg-emerald-950/40 border border-emerald-800/40 px-2 py-0.5 rounded-full">
                                    <CheckCircle2 className="w-3 h-3" /> GROUNDING VERIFIED
                                  </span>
                                ) : msg.query.status === "NO_RESULTS" ? (
                                  <span className="flex items-center gap-1 text-amber-400 bg-amber-950/40 border border-amber-800/40 px-2 py-0.5 rounded-full">
                                    <AlertCircle className="w-3 h-3" /> NO EVIDENCE MATCH
                                  </span>
                                ) : (
                                  <span className="flex items-center gap-1 text-cyan-400 bg-cyan-950/40 border border-cyan-800/40 px-2 py-0.5 rounded-full">
                                    <Activity className="w-3 h-3" /> DETERMINISTIC ANSWER
                                  </span>
                                )}
                                <span className="text-neutral-500">|</span>
                                <span className="text-neutral-400">{msg.query.execution_time_ms} ms</span>
                              </div>
                            )}
                          </div>

                          {/* Clarification Box if Ambiguous */}
                          {msg.query?.status === "AMBIGUOUS" && msg.query.clarification_needed && (
                            <div className="p-4 rounded-xl bg-amber-950/20 border border-amber-500/30 space-y-2.5">
                              <div className="flex items-center gap-2 text-amber-400 font-semibold text-xs font-mono">
                                <AlertCircle className="w-4 h-4" />
                                <span>CLARIFICATION REQUIRED</span>
                              </div>
                              <p className="text-xs text-neutral-300 leading-relaxed">
                                {msg.query.clarification_needed}
                              </p>
                              {msg.query.clarification_options && msg.query.clarification_options.length > 0 && (
                                <div className="flex flex-wrap gap-2 pt-1">
                                  {msg.query.clarification_options.map((opt) => (
                                    <button
                                      key={opt}
                                      onClick={() => handleAsk(`${msg.query?.user_query} for ${opt}`)}
                                      className="px-3 py-1 rounded-lg bg-amber-900/40 hover:bg-amber-900/70 border border-amber-500/40 text-amber-200 text-xs font-mono transition-colors cursor-pointer"
                                    >
                                      {opt}
                                    </button>
                                  ))}
                                </div>
                              )}
                            </div>
                          )}

                          {/* Grounded Natural Language Answer */}
                          <div className="p-4 rounded-xl bg-neutral-950 border border-white/10 space-y-1.5">
                            <div className="text-[11px] font-mono text-neutral-400 flex items-center gap-1.5">
                              <Sparkles className="w-3.5 h-3.5 text-cyan-400" />
                              <span>GROUNDED ANSWER</span>
                            </div>
                            <p className="text-sm text-neutral-100 font-sans leading-relaxed whitespace-pre-wrap">
                              {msg.text}
                            </p>
                          </div>

                          {/* Structured Evidence Items */}
                          {msg.query && msg.query.evidence && msg.query.evidence.length > 0 && (
                            <div className="space-y-3 pt-1">
                              <div className="flex items-center justify-between">
                                <h4 className="text-xs font-mono text-neutral-400 uppercase tracking-wider flex items-center gap-1.5">
                                  <Layers className="w-3.5 h-3.5 text-cyan-400" />
                                  <span>Visual & Structural Evidence Records ({msg.query.evidence.length})</span>
                                </h4>
                                <span className="text-[11px] font-mono text-neutral-500">
                                  Click card to navigate to source
                                </span>
                              </div>

                              <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                                {msg.query.evidence.map((evi) => (
                                  <Link
                                    key={evi.evidence_id}
                                    href={evi.action_link || "#"}
                                    className="group p-3.5 rounded-xl bg-neutral-950/70 border border-white/10 hover:border-cyan-500/40 hover:bg-neutral-900/80 transition-all flex flex-col justify-between space-y-2.5"
                                  >
                                    <div className="space-y-1.5">
                                      <div className="flex items-center justify-between">
                                        <span className="flex items-center gap-1.5 text-xs font-semibold text-white group-hover:text-cyan-300 transition-colors truncate">
                                          {getEvidenceIcon(evi.evidence_type)}
                                          <span className="truncate">{evi.title}</span>
                                        </span>
                                        <ExternalLink className="w-3.5 h-3.5 text-neutral-500 group-hover:text-cyan-400 transition-colors shrink-0 ml-1" />
                                      </div>
                                      <p className="text-xs text-neutral-400 line-clamp-2 leading-relaxed">
                                        {evi.description}
                                      </p>
                                    </div>

                                    {/* Evidence Metadata Badges */}
                                    <div className="flex flex-wrap items-center gap-1.5 pt-1 font-mono text-[10px]">
                                      {evi.class_name && (
                                        <span className="px-1.5 py-0.5 rounded bg-neutral-800 text-neutral-300 border border-white/10">
                                          class: {evi.class_name}
                                        </span>
                                      )}
                                      {evi.confidence !== null && evi.confidence !== undefined && (
                                        <span className="px-1.5 py-0.5 rounded bg-cyan-950/40 text-cyan-300 border border-cyan-800/30">
                                          conf: {(evi.confidence * 100).toFixed(0)}%
                                        </span>
                                      )}
                                      {evi.iou !== null && evi.iou !== undefined && (
                                        <span className="px-1.5 py-0.5 rounded bg-rose-950/40 text-rose-300 border border-rose-800/30">
                                          IoU: {evi.iou.toFixed(2)}
                                        </span>
                                      )}
                                      {evi.timestamp_sec !== null && evi.timestamp_sec !== undefined && (
                                        <span className="px-1.5 py-0.5 rounded bg-neutral-800 text-neutral-300 border border-white/10">
                                          t={evi.timestamp_sec.toFixed(1)}s
                                        </span>
                                      )}
                                      {evi.track_id !== null && evi.track_id !== undefined && (
                                        <span className="px-1.5 py-0.5 rounded bg-purple-950/40 text-purple-300 border border-purple-800/30">
                                          track #{evi.track_id}
                                        </span>
                                      )}
                                    </div>
                                  </Link>
                                ))}
                              </div>
                            </div>
                          )}

                          {/* Query Provenance Toggle */}
                          {msg.query && (
                            <div className="pt-1 border-t border-white/10">
                              <button
                                onClick={() =>
                                  setShowProvenanceFor(
                                    showProvenanceFor === msg.id ? null : msg.id
                                  )
                                }
                                className="text-[11px] font-mono text-neutral-400 hover:text-cyan-300 flex items-center gap-1 transition-colors cursor-pointer"
                              >
                                <Info className="w-3.5 h-3.5" />
                                <span>
                                  {showProvenanceFor === msg.id
                                    ? "Hide Query Provenance & DSL"
                                    : "Show Query Provenance & DSL"}
                                </span>
                              </button>

                              {showProvenanceFor === msg.id && (
                                <div className="mt-3 p-3.5 rounded-xl bg-neutral-950 border border-white/10 space-y-2.5 text-xs font-mono">
                                  <div>
                                    <span className="text-neutral-500 text-[10px] block mb-1">
                                      RESOLVED TARGETS:
                                    </span>
                                    <div className="p-2 rounded bg-neutral-900 border border-white/10 text-neutral-300">
                                      {Object.keys(msg.query.target).length > 0 ? (
                                        JSON.stringify(msg.query.target, null, 2)
                                      ) : (
                                        <span className="text-neutral-500">None</span>
                                      )}
                                    </div>
                                  </div>

                                  <div>
                                    <span className="text-neutral-500 text-[10px] block mb-1">
                                      ACTIVE FILTERS:
                                    </span>
                                    <div className="p-2 rounded bg-neutral-900 border border-white/10 text-neutral-300">
                                      {Object.keys(msg.query.filters).length > 0 ? (
                                        JSON.stringify(msg.query.filters, null, 2)
                                      ) : (
                                        <span className="text-neutral-500">None</span>
                                      )}
                                    </div>
                                  </div>

                                  <div>
                                    <span className="text-neutral-500 text-[10px] block mb-1">
                                      REPRODUCIBILITY HASH:
                                    </span>
                                    <div className="p-2 rounded bg-neutral-900 border border-white/10 text-cyan-300 break-all">
                                      {msg.query.reproducibility_hash}
                                    </div>
                                  </div>
                                </div>
                              )}
                            </div>
                          )}
                        </div>
                      </div>
                    )}
                  </div>
                ))}

                {/* Thinking / Synthesizing Loading State */}
                {loading && (
                  <div className="flex items-start gap-2.5">
                    <div className="w-8 h-8 rounded-xl bg-cyan-900/40 border border-cyan-500/30 flex items-center justify-center text-cyan-300 shrink-0 animate-spin">
                      <RefreshCw className="w-4 h-4" />
                    </div>
                    <div className="rounded-2xl rounded-tl-sm bg-neutral-900/90 border border-white/10 p-4 text-xs font-mono text-neutral-400 flex items-center gap-2">
                      <span className="animate-pulse">Synthesizing grounded answer from VisionForge evidence...</span>
                    </div>
                  </div>
                )}
                <div ref={messagesEndRef} />
              </div>
            )}
          </div>

          {/* Error Banner if Query Failed */}
          {errorMessage && (
            <div className="p-3.5 rounded-xl bg-rose-950/40 border border-rose-500/40 text-rose-200 text-xs flex items-center justify-between gap-3">
              <div className="flex items-center gap-2">
                <AlertCircle className="w-4 h-4 shrink-0 text-rose-400" />
                <span>{errorMessage}</span>
              </div>
              <Button
                variant="secondary"
                size="sm"
                onClick={() => checkHealth()}
                className="h-6 text-[11px] px-2 text-rose-200 border-rose-800 hover:bg-rose-900/50"
              >
                Retry
              </Button>
            </div>
          )}

          {/* Interactive Question Input Box */}
          <div className="rounded-2xl bg-neutral-900/90 border border-white/15 p-3.5 shadow-2xl focus-within:border-cyan-500/60 transition-all space-y-2">
            <div className="flex items-center gap-2.5">
              <Search className="w-4 h-4 text-neutral-400 shrink-0 ml-1" />
              <textarea
                ref={textareaRef}
                value={inputText}
                onChange={(e) => setInputText(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && !e.shiftKey) {
                    e.preventDefault();
                    if (!loading && inputText.trim()) {
                      handleAsk();
                    }
                  }
                }}
                rows={1}
                placeholder="Ask about detections, events, failure modes, models, or datasets (Enter to send, Shift+Enter for newline)..."
                className="w-full bg-transparent border-none text-white text-sm placeholder-neutral-500 focus:outline-none resize-none min-h-[36px] max-h-[120px] py-1.5 font-sans"
                disabled={loading}
              />
              <Button
                variant="primary"
                size="sm"
                onClick={() => handleAsk()}
                disabled={loading || !inputText.trim()}
                icon={<Send className="w-3.5 h-3.5" />}
                className="shrink-0 bg-cyan-600 hover:bg-cyan-500 text-black font-semibold shadow-md shadow-cyan-900/30 px-4"
              >
                {loading ? "Verifying..." : "Ask"}
              </Button>
            </div>
          </div>
        </div>

        {/* Right Sidebar: Query History & Helpful Context (4 cols) */}
        <div className="lg:col-span-4 space-y-6">
          {/* Query History Panel */}
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
                disabled={historyLoading}
                className="text-[10px] font-mono h-6 px-2 text-neutral-400 hover:text-white"
              >
                {historyLoading ? "..." : "Refresh"}
              </Button>
            </div>

            {history.length === 0 ? (
              <div className="text-center py-6 text-xs text-neutral-500">
                No previous queries recorded in this session.
              </div>
            ) : (
              <div className="space-y-2.5 max-h-[380px] overflow-y-auto pr-1 custom-scrollbar">
                {history.map((item) => (
                  <div
                    key={item.query_id}
                    className="p-3 rounded-xl bg-neutral-950/70 border border-white/10 hover:border-cyan-500/30 transition-all space-y-2"
                  >
                    <div className="flex items-center justify-between">
                      <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-neutral-800 text-cyan-300 border border-white/10">
                        {item.query_type}
                      </span>
                      <span className="text-[10px] font-mono text-neutral-400">
                        {item.results_count} results
                      </span>
                    </div>

                    <p className="text-xs text-neutral-200 line-clamp-2 font-medium">
                      "{item.user_query}"
                    </p>

                    <div className="flex items-center justify-between pt-1 text-[10px] font-mono text-neutral-500">
                      {isMounted && (
                        <span>
                          {new Date(item.created_timestamp).toLocaleTimeString([], {
                            hour: "2-digit",
                            minute: "2-digit",
                          })}
                        </span>
                      )}
                      <button
                        onClick={() => handleReplay(item.query_id)}
                        disabled={loading}
                        className="text-cyan-400 hover:text-cyan-300 flex items-center gap-1 font-sans cursor-pointer transition-colors"
                      >
                        <RotateCcw className="w-3 h-3" /> Replay
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* Factual Grounding Guidance Card */}
          <div className="rounded-2xl bg-neutral-900/80 border border-white/15 p-5 space-y-3 shadow-xl">
            <div className="flex items-center gap-2 border-b border-white/10 pb-3">
              <ShieldCheck className="w-4 h-4 text-emerald-400" />
              <h3 className="text-xs font-semibold text-white tracking-tight uppercase font-mono">
                Grounded Reasoning Policy
              </h3>
            </div>
            <ul className="space-y-2 text-xs text-neutral-300 leading-relaxed">
              <li className="flex items-start gap-2">
                <span className="text-emerald-400 shrink-0 font-bold">✓</span>
                <span>All answers derive strictly from verified detections, tracks, temporal zones, datasets, and experiment runs.</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-emerald-400 shrink-0 font-bold">✓</span>
                <span>If evidence does not exist for a claim, the assistant explicitly reports no matching records found.</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-emerald-400 shrink-0 font-bold">✓</span>
                <span>Deterministic reproducibility hash is generated for every query turn.</span>
              </li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}
