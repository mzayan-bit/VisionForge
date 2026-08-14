"use client";

import React, { useEffect, useState } from "react";
import Link from "next/link";
import {
  Activity,
  AlertCircle,
  ArrowRight,
  BarChart2,
  CheckCircle2,
  ChevronRight,
  Clock,
  Compass,
  Cpu,
  Database,
  Download,
  Eye,
  FileText,
  Filter,
  FlaskConical,
  HelpCircle,
  History,
  Layers,
  MapPin,
  MessageSquare,
  Move,
  Play,
  Pause,
  Plus,
  RefreshCw,
  Search,
  Settings,
  ShieldCheck,
  Sparkles,
  Sliders,
  Tag,
  Trash2,
  Video,
  Zap,
} from "lucide-react";
import { PageHeader } from "@/components/ui/PageHeader";
import { Button } from "@/components/ui/Button";

// ─── Interfaces ───────────────────────────────────────────────────

interface TrajectoryPoint {
  frame_index: number;
  timestamp_sec: number;
  x_center_px: number;
  y_center_px: number;
  norm_x: number;
  norm_y: number;
  width_px: number;
  height_px: number;
  bbox: number[];
}

interface Track {
  track_id: number;
  class_name: string;
  first_frame: number;
  last_frame: number;
  first_timestamp_sec: number;
  last_timestamp_sec: number;
  visibility_duration_sec: number;
  avg_confidence: number;
  min_confidence: number;
  max_confidence: number;
  total_distance_px: number;
  avg_speed_px_per_sec: number;
  status: string;
  trajectory: TrajectoryPoint[];
  detections_count: number;
}

interface TemporalAnalytics {
  total_tracks: number;
  tracks_by_class: Record<string, number>;
  avg_track_duration_sec: number;
  longest_track_duration_sec: number;
  avg_pixel_movement_px: number;
  active_objects_over_time: { second: number; active_count: number }[];
  detections_over_time: { second: number; detection_count: number }[];
}

interface VideoInferenceRun {
  run_id: string;
  video_id: string;
  model_id: string;
  tracker_name: string;
  sampling_config: {
    mode: string;
    sample_interval: number;
    total_sampled_frames: number;
  };
  timestamp: string;
  status: string;
  duration_sec: number;
  processed_frames: number;
  total_detections: number;
  total_tracks: number;
  tracks: Track[];
  analytics: TemporalAnalytics;
  processing_fps: number;
  inference_latency_ms: number;
  tracking_latency_ms: number;
}

interface RegionOfInterest {
  region_id: string;
  video_id: string;
  name: string;
  shape_type: "RECTANGLE" | "POLYGON";
  coordinates: number[][];
  coordinate_system: "PIXEL" | "NORMALIZED";
  color: string;
}

interface TemporalEvent {
  event_id: string;
  run_id: string;
  video_id: string;
  event_type: string;
  start_timestamp_sec: number;
  end_timestamp_sec: number;
  duration_sec: number;
  source_track_ids: number[];
  source_frame_range: number[];
  reliability: "HIGH" | "MEDIUM" | "LOW";
  event_params: Record<string, any>;
  description: string;
}

interface EventEvidence {
  event_id: string;
  frame_before_idx: number;
  event_frame_idx: number;
  frame_after_idx: number;
  highlight_track_ids: number[];
  highlight_region_id?: string;
  snapshot_notes: string;
}

interface QueryEvidenceItem {
  event_id?: string;
  track_id?: number;
  timestamp_sec: number;
  frame_idx: number;
  region_id?: string;
  description: string;
  action_link: string;
}

interface VisualQueryResult {
  query_id: string;
  original_query: string;
  structured_query: Record<string, any>;
  status: "SUCCESS" | "AMBIGUOUS" | "UNSUPPORTED" | "VALIDATION_ERROR";
  result_type: string;
  records: Record<string, any>[];
  summary: string;
  evidence: QueryEvidenceItem[];
  interpretation_explanation: string;
  interpretation_time_ms: number;
  execution_time_ms: number;
  total_query_time_ms: number;
  source_run_id: string;
  reproducibility_hash: string;
  created_at: string;
}

interface QueryHistoryItem {
  query_id: string;
  original_query: string;
  query_type: string;
  run_id: string;
  status: string;
  results_count: number;
  total_query_time_ms: number;
  created_at: string;
}

export default function VideoLabPage() {
  // Video Controls State
  const [videoId, setVideoId] = useState<string>("sample_traffic_01");
  const [modelId, setModelId] = useState<string>("yolo11s.pt");
  const [trackerName, setTrackerName] = useState<string>("ByteTrack");
  const [samplingMode, setSamplingMode] = useState<string>("EVERY_2ND_FRAME");

  // Playback & Video Player State
  const [isPlaying, setIsPlaying] = useState<boolean>(false);
  const [currentTimeSec, setCurrentTimeSec] = useState<number>(0.0);
  const [playbackSpeed, setPlaybackSpeed] = useState<number>(1.0);

  // Active Run & Selection State
  const [currentRun, setCurrentRun] = useState<VideoInferenceRun | null>(null);
  const [selectedTrackId, setSelectedTrackId] = useState<number | null>(null);
  const [showTrajectory, setShowTrajectory] = useState<boolean>(true);
  const [loading, setLoading] = useState<boolean>(false);

  // Temporal Event Intelligence State
  const [regions, setRegions] = useState<RegionOfInterest[]>([]);
  const [events, setEvents] = useState<TemporalEvent[]>([]);
  const [selectedEventId, setSelectedEventId] = useState<string | null>(null);
  const [inspectEvidenceModal, setInspectEvidenceModal] = useState<EventEvidence | null>(null);
  const [eventFilterType, setEventFilterType] = useState<string>("ALL");
  const [showAddRegionModal, setShowAddRegionModal] = useState<boolean>(false);
  const [newRegionName, setNewRegionName] = useState<string>("Loading Zone A");

  // Visual Query Layer State
  const [userQuestion, setUserQuestion] = useState<string>("");
  const [queryResult, setQueryResult] = useState<VisualQueryResult | null>(null);
  const [queryHistory, setQueryHistory] = useState<QueryHistoryItem[]>([]);
  const [askingQuery, setAskingQuery] = useState<boolean>(false);
  const [showQueryBuilderModal, setShowQueryBuilderModal] = useState<boolean>(false);
  const [showHistoryDrawer, setShowHistoryDrawer] = useState<boolean>(false);

  // Query Builder Form State
  const [qbQueryType, setQbQueryType] = useState<string>("EVENT_SEARCH");
  const [qbEventType, setQbEventType] = useState<string>("OBJECT_ENTERED_REGION");
  const [qbClass, setQbClass] = useState<string>("person");
  const [qbRegion, setQbRegion] = useState<string>("Loading Zone A");
  const [qbMinDuration, setQbMinDuration] = useState<number>(3.0);

  useEffect(() => {
    handleRunVideoInference();
  }, []);

  // Timer loop for video playback simulation
  useEffect(() => {
    let interval: NodeJS.Timeout;
    if (isPlaying && currentRun) {
      interval = setInterval(() => {
        setCurrentTimeSec((prev) => {
          const next = prev + 0.1 * playbackSpeed;
          if (next >= currentRun.duration_sec) {
            setIsPlaying(false);
            return 0;
          }
          return next;
        });
      }, 100);
    }
    return () => clearInterval(interval);
  }, [isPlaying, currentRun, playbackSpeed]);

  const handleRunVideoInference = async () => {
    setLoading(true);
    try {
      const res = await fetch("/api/v1/video/runs", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          video_id: videoId,
          model_id: modelId,
          sampling_mode: samplingMode,
          custom_stride: samplingMode === "EVERY_FRAME" ? 1 : samplingMode === "EVERY_5TH_FRAME" ? 5 : 2,
        }),
      });

      if (res.ok) {
        const data: VideoInferenceRun = await res.json();
        setCurrentRun(data);
        setCurrentTimeSec(0.0);
        if (data.tracks.length > 0) {
          setSelectedTrackId(data.tracks[0].track_id);
        }

        await fetchRegions(data.video_id);
        await handleGenerateEvents(data.run_id);
        await fetchQueryHistory();
      }
    } catch (err) {
      console.error("Failed to run video inference pipeline:", err);
    } finally {
      setLoading(false);
    }
  };

  const fetchRegions = async (vidId: string) => {
    try {
      const res = await fetch(`/api/v1/events/regions?video_id=${vidId}`);
      if (res.ok) {
        const data: RegionOfInterest[] = await res.json();
        setRegions(data);
      }
    } catch (err) {
      console.error("Failed to fetch regions:", err);
    }
  };

  const handleAddDefaultRegion = async () => {
    if (!currentRun) return;
    try {
      const res = await fetch("/api/v1/events/regions", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          video_id: currentRun.video_id,
          name: newRegionName,
          shape_type: "RECTANGLE",
          coordinates: [
            [200.0, 150.0],
            [1200.0, 700.0],
          ],
          color: "#3b82f6",
        }),
      });

      if (res.ok) {
        await fetchRegions(currentRun.video_id);
        await handleGenerateEvents(currentRun.run_id);
        setShowAddRegionModal(false);
      }
    } catch (err) {
      console.error("Failed to create region:", err);
    }
  };

  const handleDeleteRegion = async (regId: string) => {
    try {
      await fetch(`/api/v1/events/regions/${regId}`, { method: "DELETE" });
      if (currentRun) {
        await fetchRegions(currentRun.video_id);
        await handleGenerateEvents(currentRun.run_id);
      }
    } catch (err) {
      console.error("Failed to delete region:", err);
    }
  };

  const handleGenerateEvents = async (runId: string) => {
    try {
      const res = await fetch("/api/v1/events/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          run_id: runId,
          config: {
            dwell_threshold_sec: 3.0,
            proximity_threshold_px: 100.0,
          },
        }),
      });

      if (res.ok) {
        const evts: TemporalEvent[] = await res.json();
        setEvents(evts);
        if (evts.length > 0) {
          setSelectedEventId(evts[0].event_id);
        }
      }
    } catch (err) {
      console.error("Failed to generate temporal events:", err);
    }
  };

  const handleAskQuery = async (queryStr?: string) => {
    const textToSubmit = queryStr || userQuestion;
    if (!textToSubmit || !currentRun) return;

    setAskingQuery(true);
    try {
      const res = await fetch("/api/v1/query/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query_text: textToSubmit,
          run_id: currentRun.run_id,
        }),
      });

      if (res.ok) {
        const result: VisualQueryResult = await res.json();
        setQueryResult(result);
        setUserQuestion(textToSubmit);
        await fetchQueryHistory();

        // If evidence exists, auto seek to first evidence item
        if (result.evidence.length > 0) {
          const ev = result.evidence[0];
          setCurrentTimeSec(ev.timestamp_sec);
          if (ev.track_id !== undefined) {
            setSelectedTrackId(ev.track_id);
          }
        }
      }
    } catch (err) {
      console.error("Failed to execute visual query:", err);
    } finally {
      setAskingQuery(false);
    }
  };

  const handleRunStructuredQueryBuilder = async () => {
    if (!currentRun) return;
    setAskingQuery(true);
    try {
      const res = await fetch("/api/v1/query/execute", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query: {
            query_id: `vq_builder_${Date.now()}`,
            run_id: currentRun.run_id,
            query_type: qbQueryType,
            event_type: qbEventType,
            object_class: qbClass,
            region_name: qbRegion,
            min_duration_sec: qbMinDuration,
            original_text: `Visual Query: ${qbEventType} in ${qbRegion}`,
          },
        }),
      });

      if (res.ok) {
        const result: VisualQueryResult = await res.json();
        setQueryResult(result);
        setShowQueryBuilderModal(false);
        await fetchQueryHistory();
      }
    } catch (err) {
      console.error("Failed to run query builder:", err);
    } finally {
      setAskingQuery(false);
    }
  };

  const fetchQueryHistory = async () => {
    try {
      const res = await fetch("/api/v1/query/history");
      if (res.ok) {
        const history: QueryHistoryItem[] = await res.json();
        setQueryHistory(history);
      }
    } catch (err) {
      console.error("Failed to fetch query history:", err);
    }
  };

  const handleInspectEvidence = async (eventId: string) => {
    try {
      const res = await fetch(`/api/v1/events/${eventId}/evidence`);
      if (res.ok) {
        const evidence: EventEvidence = await res.json();
        setInspectEvidenceModal(evidence);
      }
    } catch (err) {
      console.error("Failed to fetch event evidence:", err);
    }
  };

  const selectedTrack = currentRun?.tracks.find((t) => t.track_id === selectedTrackId);
  const selectedEvent = events.find((e) => e.event_id === selectedEventId);

  const activeTracksAtCurrentTime = currentRun?.tracks.filter(
    (t) =>
      currentTimeSec >= t.first_timestamp_sec && currentTimeSec <= t.last_timestamp_sec + 0.5
  ) || [];

  const filteredEvents = events.filter((e) => {
    if (eventFilterType !== "ALL" && e.event_type !== eventFilterType) return false;
    return true;
  });

  const SAMPLE_QUESTIONS = [
    "Which objects entered Loading Zone A?",
    "How many people were present at 5 seconds?",
    "Which track stayed longest in Loading Zone A?",
    "Show events involving Track 1.",
    "Which objects became close?",
    "Show dwell events longer than 3 seconds.",
  ];

  return (
    <div className="flex flex-col min-h-screen bg-[#0a0a0a] text-neutral-200 font-inter">
      {/* Page Header */}
      <PageHeader
        title="Video Lab & Ask VisionForge Query Layer"
        description="Structured Computer Vision Query Layer: Natural Language Questions, Verified Visual Evidence & Query Builder"
        breadcrumbs={["VisionForge", "Video Lab", "Ask VisionForge"]}
        actions={
          <div className="flex items-center gap-2">
            <Button
              variant="secondary"
              icon={<History className="w-4 h-4 text-purple-400" />}
              onClick={() => setShowHistoryDrawer(true)}
            >
              Query History ({queryHistory.length})
            </Button>

            <Button
              variant="secondary"
              icon={<Sliders className="w-4 h-4 text-blue-400" />}
              onClick={() => setShowQueryBuilderModal(true)}
            >
              Visual Query Builder
            </Button>

            <Button
              variant="primary"
              icon={<Video className="w-4 h-4" />}
              onClick={handleRunVideoInference}
              disabled={loading}
            >
              {loading ? "Processing..." : "Run Video Pipeline"}
            </Button>
          </div>
        }
      />

      <div className="p-6 space-y-6 flex-1">
        {/* Ask VisionForge Natural Language Search Panel */}
        <div className="bg-[#121212] border border-blue-500/30 rounded-xl p-5 space-y-4 shadow-xl">
          <div className="flex justify-between items-center border-b border-white/10 pb-3">
            <h3 className="text-sm font-semibold uppercase tracking-wider text-blue-400 flex items-center gap-2">
              <MessageSquare className="w-4.5 h-4.5 text-blue-400" />
              Ask VisionForge Visual Query Engine
            </h3>
            <span className="text-[10px] font-mono text-neutral-500 bg-[#1a1a1a] px-2 py-1 rounded border border-white/5">
              Read-Only Security Guarantee | Evidence Backed
            </span>
          </div>

          {/* Search Bar Input */}
          <div className="flex items-center gap-3">
            <div className="relative flex-1">
              <Search className="w-4 h-4 absolute left-3.5 top-1/2 -translate-y-1/2 text-neutral-500" />
              <input
                type="text"
                value={userQuestion}
                onChange={(e) => setUserQuestion(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && handleAskQuery()}
                placeholder="Ask a question (e.g. 'Which objects entered Loading Zone A?', 'How many people at 5 seconds?')"
                className="w-full bg-[#181818] border border-white/10 rounded-xl pl-10 pr-4 py-2.5 text-sm text-white focus:outline-none focus:border-blue-500 font-mono"
              />
            </div>
            <Button
              variant="primary"
              icon={<Zap className="w-4 h-4" />}
              onClick={() => handleAskQuery()}
              disabled={askingQuery || !userQuestion}
            >
              {askingQuery ? "Analyzing..." : "Ask Question"}
            </Button>
          </div>

          {/* Quick Question Chips */}
          <div className="flex flex-wrap items-center gap-2 pt-1">
            <span className="text-[10px] font-mono text-neutral-500 flex items-center gap-1">
              <Sparkles className="w-3 h-3 text-amber-400" /> Quick Queries:
            </span>
            {SAMPLE_QUESTIONS.map((q, idx) => (
              <button
                key={idx}
                onClick={() => handleAskQuery(q)}
                className="text-[10px] font-mono px-2.5 py-1 rounded-full bg-[#1a1a1a] hover:bg-blue-600/20 text-neutral-300 hover:text-blue-300 border border-white/5 hover:border-blue-500/40 transition-all"
              >
                {q}
              </button>
            ))}
          </div>

          {/* Interpreted Query & Result Display */}
          {queryResult && (
            <div className="mt-4 bg-[#161616] border border-white/10 rounded-xl p-4 space-y-3 font-mono text-xs">
              <div className="flex flex-wrap justify-between items-center border-b border-white/10 pb-2 gap-2">
                <div className="flex items-center gap-2">
                  <span
                    className={`px-2 py-0.5 rounded text-[10px] font-bold ${
                      queryResult.status === "SUCCESS"
                        ? "bg-emerald-500/20 text-emerald-400 border border-emerald-500/30"
                        : queryResult.status === "AMBIGUOUS"
                        ? "bg-amber-500/20 text-amber-400 border border-amber-500/30"
                        : "bg-purple-500/20 text-purple-400 border border-purple-500/30"
                    }`}
                  >
                    STATUS: {queryResult.status}
                  </span>
                  <span className="text-neutral-400 text-[11px]">
                    Query ID: {queryResult.query_id}
                  </span>
                </div>

                <div className="text-[10px] text-neutral-500">
                  Latency: {queryResult.total_query_time_ms}ms | Records: {queryResult.records.length}
                </div>
              </div>

              {/* Interpretation Explanation Badge */}
              <div className="bg-[#1c1c1c] p-2.5 rounded border border-white/5 text-blue-300 text-[11px]">
                <span className="font-bold text-neutral-400">Interpreted Query DSL:</span>{" "}
                {queryResult.interpretation_explanation}
              </div>

              {/* Natural Language Summary Answer */}
              <div className="p-3 bg-blue-950/20 border border-blue-500/30 rounded-lg text-white font-bold text-xs leading-relaxed">
                Answer: {queryResult.summary}
              </div>

              {/* Evidence Stream Cards */}
              {queryResult.evidence.length > 0 && (
                <div className="space-y-2 pt-2">
                  <div className="text-[11px] font-bold text-neutral-400 uppercase tracking-wider flex items-center gap-1.5">
                    <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" />
                    Verified Visual Evidence ({queryResult.evidence.length} sources)
                  </div>

                  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-2">
                    {queryResult.evidence.map((ev, idx) => (
                      <div
                        key={idx}
                        className="bg-[#1a1a1a] p-3 rounded-lg border border-white/5 space-y-2 hover:border-blue-500/40 transition-all"
                      >
                        <div className="flex justify-between items-center text-[10px]">
                          <span className="text-blue-400 font-bold">t = {ev.timestamp_sec.toFixed(1)}s</span>
                          {ev.track_id !== undefined && (
                            <span className="text-purple-400 font-bold">Track #{ev.track_id}</span>
                          )}
                        </div>

                        <p className="text-[10px] text-neutral-300 line-clamp-2 leading-relaxed">
                          {ev.description}
                        </p>

                        <button
                          onClick={() => {
                            setCurrentTimeSec(ev.timestamp_sec);
                            if (ev.track_id !== undefined) setSelectedTrackId(ev.track_id);
                          }}
                          className="w-full text-center py-1 rounded bg-blue-600/20 hover:bg-blue-600/30 text-blue-300 text-[10px] border border-blue-500/30 transition-all font-bold"
                        >
                          [ View Evidence at t={ev.timestamp_sec.toFixed(1)}s ]
                        </button>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Video Player & Event Stream Layout */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Left Column: Video Player & Region Overlay */}
          <div className="lg:col-span-2 space-y-6">
            <div className="bg-[#121212] border border-white/10 rounded-xl overflow-hidden space-y-4 p-4">
              <div className="flex justify-between items-center border-b border-white/10 pb-3">
                <h3 className="text-xs font-semibold uppercase tracking-wider text-neutral-400 flex items-center gap-2">
                  <Video className="w-4 h-4 text-cyan-400" />
                  Video Canvas with Region & Track Overlay
                </h3>
                <span className="text-xs text-neutral-500 font-mono">
                  Timestamp: {currentTimeSec.toFixed(1)}s / {currentRun?.duration_sec.toFixed(1) || "10.0"}s
                </span>
              </div>

              {/* Video Overlay Screen */}
              <div className="relative aspect-video bg-[#080808] border border-white/10 rounded-lg overflow-hidden flex flex-col items-center justify-center">
                <div className="absolute inset-0 bg-[radial-gradient(#1f1f1f_1px,transparent_1px)] [background-size:16px_16px] opacity-40" />

                <svg className="absolute inset-0 w-full h-full pointer-events-none">
                  {/* Region ROI Overlays */}
                  {regions.map((reg) => (
                    <g key={`reg_${reg.region_id}`}>
                      <rect
                        x={`${(reg.coordinates[0][0] / 1920) * 100}%`}
                        y={`${(reg.coordinates[0][1] / 1080) * 100}%`}
                        width={`${((reg.coordinates[1][0] - reg.coordinates[0][0]) / 1920) * 100}%`}
                        height={`${((reg.coordinates[1][1] - reg.coordinates[0][1]) / 1080) * 100}%`}
                        fill="rgba(59, 130, 246, 0.08)"
                        stroke={reg.color}
                        strokeWidth="2"
                        strokeDasharray="6 3"
                        rx="6"
                      />
                      <foreignObject
                        x={`${(reg.coordinates[0][0] / 1920) * 100}%`}
                        y={`${(reg.coordinates[0][1] / 1080) * 100 + 1}%`}
                        width="160"
                        height="24"
                      >
                        <div className="text-[10px] font-mono font-bold px-2 py-0.5 rounded bg-blue-600/40 text-blue-300 border border-blue-500/40 w-fit">
                          Region: {reg.name}
                        </div>
                      </foreignObject>
                    </g>
                  ))}

                  {/* Selected Track Trajectory Line */}
                  {showTrajectory && selectedTrack && (
                    <polyline
                      points={selectedTrack.trajectory
                        .filter((pt) => pt.timestamp_sec <= currentTimeSec)
                        .map((pt) => `${pt.norm_x * 100}% ${pt.norm_y * 100}%`)
                        .join(", ")}
                      fill="none"
                      stroke="#3b82f6"
                      strokeWidth="2.5"
                      strokeDasharray="4 2"
                    />
                  )}

                  {/* Active Track Bounding Boxes */}
                  {activeTracksAtCurrentTime.map((track) => {
                    const latestPt = track.trajectory.reduce(
                      (prev, curr) => (curr.timestamp_sec <= currentTimeSec ? curr : prev),
                      track.trajectory[0]
                    );

                    const isSelected = selectedTrackId === track.track_id;

                    return (
                      <g key={`t_overlay_${track.track_id}`}>
                        <rect
                          x={`${latestPt.bbox[0] / 19.2}%`}
                          y={`${latestPt.bbox[1] / 10.8}%`}
                          width={`${latestPt.width_px / 19.2}%`}
                          height={`${latestPt.height_px / 10.8}%`}
                          fill={isSelected ? "rgba(59, 130, 246, 0.25)" : "rgba(168, 85, 247, 0.15)"}
                          stroke={isSelected ? "#3b82f6" : "#a855f7"}
                          strokeWidth={isSelected ? "3" : "1.5"}
                          rx="4"
                        />
                        <foreignObject
                          x={`${latestPt.bbox[0] / 19.2}%`}
                          y={`${latestPt.bbox[1] / 10.8 - 6}%`}
                          width="140"
                          height="24"
                        >
                          <div
                            className={`text-[10px] font-mono font-bold px-1.5 py-0.5 rounded shadow w-fit ${
                              isSelected ? "bg-blue-600 text-white" : "bg-purple-600 text-white"
                            }`}
                          >
                            Track #{track.track_id} ({track.class_name})
                          </div>
                        </foreignObject>
                      </g>
                    );
                  })}
                </svg>

                <div className="z-10 text-center space-y-1">
                  <div className="text-xs font-mono text-neutral-400">
                    Video Stream ({currentRun?.processed_frames || 0} sampled frames)
                  </div>
                  <div className="text-[10px] text-neutral-500 font-mono">
                    Active Regions: {regions.length} | Detected Events: {events.length}
                  </div>
                </div>
              </div>

              {/* Scrubber & Controls */}
              <div className="space-y-3 pt-2">
                <input
                  type="range"
                  min={0}
                  max={currentRun?.duration_sec || 10.0}
                  step={0.1}
                  value={currentTimeSec}
                  onChange={(e) => setCurrentTimeSec(parseFloat(e.target.value))}
                  className="w-full h-1.5 bg-[#1f1f1f] rounded-lg appearance-none cursor-pointer accent-blue-500"
                />

                <div className="flex justify-between items-center">
                  <div className="flex items-center gap-2">
                    <button
                      onClick={() => setIsPlaying(!isPlaying)}
                      className="p-2 rounded-lg bg-blue-600/20 text-blue-400 hover:bg-blue-600/30 border border-blue-500/30 transition-all"
                    >
                      {isPlaying ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
                    </button>

                    <div className="flex items-center gap-1 text-xs font-mono">
                      {[0.5, 1.0, 2.0].map((s) => (
                        <button
                          key={s}
                          onClick={() => setPlaybackSpeed(s)}
                          className={`px-2 py-0.5 rounded border text-[10px] ${
                            playbackSpeed === s
                              ? "bg-blue-600/20 text-blue-400 border-blue-500/30"
                              : "bg-[#181818] text-neutral-400 border-white/5"
                          }`}
                        >
                          {s}x
                        </button>
                      ))}
                    </div>
                  </div>

                  <div className="flex items-center gap-3 text-xs font-mono">
                    <label className="flex items-center gap-1.5 cursor-pointer text-neutral-400">
                      <input
                        type="checkbox"
                        checked={showTrajectory}
                        onChange={(e) => setShowTrajectory(e.target.checked)}
                        className="rounded accent-blue-500"
                      />
                      <span>Show Trajectories</span>
                    </label>

                    <span className="text-neutral-500">|</span>
                    <span className="text-emerald-400 font-bold">
                      {currentRun?.processing_fps || 0} FPS
                    </span>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Right Column: Event Timeline & Regions */}
          <div className="space-y-6">
            <div className="bg-[#121212] border border-white/10 rounded-xl overflow-hidden flex flex-col h-[560px]">
              <div className="p-4 border-b border-white/10 flex justify-between items-center bg-[#161616] shrink-0">
                <h3 className="text-xs font-semibold uppercase tracking-wider text-neutral-400 flex items-center gap-2">
                  <Clock className="w-4 h-4 text-emerald-400" />
                  Chronological Event Stream ({filteredEvents.length})
                </h3>
              </div>

              <div className="p-3 space-y-2 overflow-y-auto flex-1 font-mono text-xs">
                {filteredEvents.map((evt) => {
                  const isSelected = selectedEventId === evt.event_id;

                  return (
                    <div
                      key={evt.event_id}
                      onClick={() => {
                        setSelectedEventId(evt.event_id);
                        setCurrentTimeSec(evt.start_timestamp_sec);
                        if (evt.source_track_ids.length > 0) {
                          setSelectedTrackId(evt.source_track_ids[0]);
                        }
                      }}
                      className={`p-3 rounded-lg border cursor-pointer transition-all space-y-1.5 ${
                        isSelected
                          ? "bg-blue-600/20 border-blue-500/50 text-white"
                          : "bg-[#181818] border-white/5 hover:border-white/20 text-neutral-400"
                      }`}
                    >
                      <div className="flex justify-between items-center">
                        <span className="text-[11px] font-bold text-blue-400">
                          t={evt.start_timestamp_sec.toFixed(1)}s
                        </span>
                        <span className="text-[9px] font-bold px-1.5 py-0.5 rounded bg-neutral-800 text-neutral-300">
                          {evt.event_type}
                        </span>
                      </div>
                      <p className="text-[11px] text-neutral-300 line-clamp-2 leading-relaxed">
                        {evt.description}
                      </p>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Visual Query Builder Modal */}
      {showQueryBuilderModal && (
        <div className="fixed inset-0 bg-black/80 z-50 flex items-center justify-center p-4">
          <div className="bg-[#121212] border border-white/10 rounded-xl p-6 max-w-lg w-full space-y-4 font-mono text-xs">
            <h3 className="text-sm font-semibold text-white flex items-center gap-2">
              <Sliders className="w-4 h-4 text-blue-400" />
              Visual Query Builder (Structured DSL)
            </h3>

            <div className="space-y-3">
              <div>
                <label className="text-neutral-400 block mb-1">Query Type</label>
                <select
                  value={qbQueryType}
                  onChange={(e) => setQbQueryType(e.target.value)}
                  className="w-full bg-[#1a1a1a] border border-white/10 rounded px-3 py-2 text-white"
                >
                  <option value="EVENT_SEARCH">EVENT_SEARCH</option>
                  <option value="TRACK_SEARCH">TRACK_SEARCH</option>
                  <option value="OBJECT_COUNT">OBJECT_COUNT</option>
                  <option value="TRACK_AGGREGATION">TRACK_AGGREGATION</option>
                </select>
              </div>

              <div>
                <label className="text-neutral-400 block mb-1">Event Type</label>
                <select
                  value={qbEventType}
                  onChange={(e) => setQbEventType(e.target.value)}
                  className="w-full bg-[#1a1a1a] border border-white/10 rounded px-3 py-2 text-white"
                >
                  <option value="OBJECT_ENTERED_REGION">OBJECT_ENTERED_REGION</option>
                  <option value="OBJECT_LEFT_REGION">OBJECT_LEFT_REGION</option>
                  <option value="OBJECT_DWELLED">OBJECT_DWELLED</option>
                  <option value="OBJECT_STOPPED">OBJECT_STOPPED</option>
                </select>
              </div>

              <div>
                <label className="text-neutral-400 block mb-1">Object Class</label>
                <select
                  value={qbClass}
                  onChange={(e) => setQbClass(e.target.value)}
                  className="w-full bg-[#1a1a1a] border border-white/10 rounded px-3 py-2 text-white"
                >
                  <option value="person">person</option>
                  <option value="car">car</option>
                </select>
              </div>

              <div>
                <label className="text-neutral-400 block mb-1">Target Region</label>
                <select
                  value={qbRegion}
                  onChange={(e) => setQbRegion(e.target.value)}
                  className="w-full bg-[#1a1a1a] border border-white/10 rounded px-3 py-2 text-white"
                >
                  {regions.map((r) => (
                    <option key={r.region_id} value={r.name}>
                      {r.name}
                    </option>
                  ))}
                </select>
              </div>
            </div>

            <div className="flex justify-end gap-2 pt-3">
              <Button variant="secondary" size="sm" onClick={() => setShowQueryBuilderModal(false)}>
                Cancel
              </Button>
              <Button variant="primary" size="sm" onClick={handleRunStructuredQueryBuilder}>
                Run Structured Query
              </Button>
            </div>
          </div>
        </div>
      )}

      {/* Query History Drawer */}
      {showHistoryDrawer && (
        <div className="fixed inset-0 bg-black/80 z-50 flex items-center justify-end">
          <div className="bg-[#121212] border-l border-white/10 w-full max-w-md h-full p-6 space-y-4 font-mono text-xs overflow-y-auto">
            <div className="flex justify-between items-center border-b border-white/10 pb-3">
              <h3 className="text-sm font-semibold text-white flex items-center gap-2">
                <History className="w-4 h-4 text-purple-400" />
                Query Execution History
              </h3>
              <button onClick={() => setShowHistoryDrawer(false)} className="text-neutral-400 hover:text-white">
                ✕
              </button>
            </div>

            <div className="space-y-2">
              {queryHistory.map((item) => (
                <div
                  key={item.query_id}
                  className="p-3 bg-[#181818] border border-white/5 rounded-lg space-y-1.5 hover:border-purple-500/40 transition-all"
                >
                  <div className="flex justify-between items-center">
                    <span className="text-purple-400 font-bold">{item.query_type}</span>
                    <span className="text-[10px] text-neutral-500">{item.total_query_time_ms}ms</span>
                  </div>
                  <p className="text-white font-semibold text-xs">{item.original_query}</p>
                  <div className="flex justify-between items-center pt-1 text-[10px] text-neutral-500">
                    <span>Records: {item.results_count}</span>
                    <button
                      onClick={() => {
                        setShowHistoryDrawer(false);
                        handleAskQuery(item.original_query);
                      }}
                      className="text-blue-400 hover:underline"
                    >
                      Re-run Query
                    </button>
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
