"use client";

import React, { useEffect, useRef, useState } from "react";
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
  TrendingUp,
  Box,
  Split,
  Maximize2,
  RotateCcw,
  Sparkle,
  Target,
  Share2,
  Upload,
} from "lucide-react";
import { PageHeader } from "@/components/ui/PageHeader";
import { Button } from "@/components/ui/Button";

// ─── Interfaces ───────────────────────────────────────────────────

interface VideoMetadata {
  video_id: string;
  filename: string;
  duration_sec: number;
  fps: number;
  frame_count: number;
  width: number;
  height: number;
  codec: string;
  size_bytes: number;
  created_at: string;
}

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
  instantaneous_speed_px_s?: number;
}

interface RegionVisit {
  region_id: string;
  region_name: string;
  entered_sec: number;
  exited_sec?: number | null;
  dwell_duration_sec: number;
  visit_count: number;
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
  image_space_velocity_px_s: number;
  observation_count: number;
  gap_count: number;
  regions_visited?: RegionVisit[];
  associated_events?: string[];
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
  total_region_visits?: number;
  avg_dwell_time_sec?: number;
  events_per_minute?: number;
  active_objects_over_time: { second: number; count?: number; active_count?: number }[];
  detections_over_time: { second: number; count?: number; detection_count?: number }[];
}

interface EventEvidence {
  event_id: string;
  frame_before_idx: number;
  event_frame_idx: number;
  frame_after_idx: number;
  start_timestamp_sec: number;
  representative_timestamp_sec: number;
  end_timestamp_sec: number;
  highlight_track_ids: number[];
  highlight_region_id?: string | null;
  trigger_rule: string;
  snapshot_notes: string;
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
  reliability: string;
  event_params: Record<string, any>;
  description: string;
  trigger_rule?: string;
  evidence?: EventEvidence | null;
  created_at: string;
}

interface RegionOfInterest {
  region_id: string;
  video_id: string;
  name: string;
  shape_type: string;
  coordinates: number[][];
  coordinate_system: string;
  color: string;
  created_at: string;
}

interface VideoSession {
  session_id: string;
  video_id: string;
  video_source: string;
  duration_sec: number;
  fps: number;
  width: number;
  height: number;
  frame_count: number;
  codec: string;
  file_size_bytes: number;
  model_version: string;
  status: string;
  video_fingerprint: string;
  lineage: Record<string, any>;
  created_at: string;
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

export default function VideoLabPage() {
  // ─── State ──────────────────────────────────────────────────────────
  const [videos, setVideos] = useState<VideoMetadata[]>([]);
  const [sessions, setSessions] = useState<VideoSession[]>([]);
  const [selectedSession, setSelectedSession] = useState<VideoSession | null>(null);
  const [runs, setRuns] = useState<VideoInferenceRun[]>([]);
  const [selectedRun, setSelectedRun] = useState<VideoInferenceRun | null>(null);
  const [regions, setRegions] = useState<RegionOfInterest[]>([]);
  const [events, setEvents] = useState<TemporalEvent[]>([]);
  const [selectedTrack, setSelectedTrack] = useState<Track | null>(null);
  const [selectedEvent, setSelectedEvent] = useState<TemporalEvent | null>(null);

  // Playback & Overlay State
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const [hasVideoError, setHasVideoError] = useState<boolean>(false);
  const [currentTimeSec, setCurrentTimeSec] = useState<number>(0.0);
  const [isPlaying, setIsPlaying] = useState<boolean>(false);
  const [playbackSpeed, setPlaybackSpeed] = useState<number>(1.0);
  const [showBoxes, setShowBoxes] = useState<boolean>(true);
  const [showTrackIds, setShowTrackIds] = useState<boolean>(true);
  const [showTrajectories, setShowTrajectories] = useState<boolean>(true);
  const [showRegions, setShowRegions] = useState<boolean>(true);
  const [showEvents, setShowEvents] = useState<boolean>(true);

  // Search & Query Layer
  const [queryInput, setQueryInput] = useState<string>("");
  const [queryResult, setQueryResult] = useState<any | null>(null);
  const [isQuerying, setIsQuerying] = useState<boolean>(false);
  const [activeTab, setActiveTab] = useState<"timeline" | "tracks" | "regions" | "query" | "lineage">("timeline");

  // Region Creation Modal
  const [isRegionModalOpen, setIsRegionModalOpen] = useState<boolean>(false);
  const [newRegionName, setNewRegionName] = useState<string>("Corridor B");
  const [newRegionColor, setNewRegionColor] = useState<string>("#3b82f6");

  // Video Comparison Modal
  const [isCompareModalOpen, setIsCompareModalOpen] = useState<boolean>(false);
  const [compareResult, setCompareResult] = useState<any | null>(null);

  // Video Upload Modal
  const [isUploadModalOpen, setIsUploadModalOpen] = useState<boolean>(false);
  const [uploadFile, setUploadFile] = useState<File | null>(null);
  const [isUploading, setIsUploading] = useState<boolean>(false);
  const [uploadError, setUploadError] = useState<string | null>(null);

  const handleUploadVideo = async () => {
    if (!uploadFile) return;
    setIsUploading(true);
    setUploadError(null);
    try {
      const formData = new FormData();
      formData.append("file", uploadFile);

      const res = await fetch("/api/v1/video/upload", {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.detail || errData.message || "Video upload failed");
      }

      const meta = await res.json();
      const videoId = meta.video_id || meta.data?.video_id;
      if (!videoId) {
        throw new Error("No video ID returned from server.");
      }

      // Automatically trigger tracking run with correct schema
      const runRes = await fetch("/api/v1/video/runs", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          video_id: videoId,
          model_id: "yolo11s.pt",
          sampling_mode: "EVERY_2ND_FRAME",
          custom_stride: 2,
        }),
      });

      let newRun: any = null;
      if (runRes.ok) {
        newRun = await runRes.json();
      }

      await fetchSessions();
      const runsRes = await fetch("/api/v1/video/runs");
      if (runsRes.ok) {
        const allRuns = await runsRes.json();
        setRuns(allRuns);
        if (newRun) {
          setSelectedRun(newRun);
          fetchRegions(videoId);
          fetchEvents(newRun.run_id);
        } else if (allRuns.length > 0) {
          setSelectedRun(allRuns[0]);
          fetchRegions(allRuns[0].video_id);
          fetchEvents(allRuns[0].run_id);
        }
      }

      setIsUploadModalOpen(false);
      setUploadFile(null);
    } catch (err: any) {
      setUploadError(err.message || "Failed to upload and analyze video");
    } finally {
      setIsUploading(false);
    }
  };

  // Initial Load
  useEffect(() => {
    fetchVideos();
    fetchSessions();
    fetchRuns();
  }, []);

  const fetchVideos = async () => {
    try {
      const res = await fetch("/api/v1/video/videos");
      if (res.ok) {
        const data = await res.json();
        setVideos(data);
      }
    } catch (e) {
      console.warn("Using fallback videos:", e);
    }
  };

  const fetchSessions = async () => {
    try {
      const res = await fetch("/api/v1/video/sessions");
      if (res.ok) {
        const data = await res.json();
        setSessions(data);
        if (data.length > 0) setSelectedSession(data[0]);
      }
    } catch (e) {
      console.warn("Using fallback sessions:", e);
    }
  };

  const fetchRuns = async () => {
    try {
      const res = await fetch("/api/v1/video/runs");
      if (res.ok) {
        const data = await res.json();
        setRuns(data);
        if (data.length > 0) {
          setSelectedRun(data[0]);
          fetchRegions(data[0].video_id);
          fetchEvents(data[0].run_id);
        }
      }
    } catch (e) {
      console.warn("Using fallback runs:", e);
    }
  };

  const fetchRegions = async (videoId: string) => {
    try {
      const res = await fetch(`/api/v1/events/regions?video_id=${videoId}`);
      if (res.ok) {
        const data = await res.json();
        setRegions(data);
      }
    } catch (e) {
      console.warn("Failed fetching regions:", e);
    }
  };

  const fetchEvents = async (runId: string) => {
    try {
      const res = await fetch(`/api/v1/events/runs/${runId}`);
      if (res.ok) {
        const data = await res.json();
        setEvents(data);
      }
    } catch (e) {
      console.warn("Failed fetching events:", e);
    }
  };

  // Playback timer ticker (Fallback when native video is unavailable or loading)
  useEffect(() => {
    let interval: any = null;
    if (isPlaying && (hasVideoError || !videoRef.current) && selectedRun) {
      interval = setInterval(() => {
        setCurrentTimeSec((prev) => {
          const next = prev + 0.1 * playbackSpeed;
          if (next >= selectedRun.duration_sec) {
            setIsPlaying(false);
            return 0.0;
          }
          return parseFloat(next.toFixed(2));
        });
      }, 100);
    } else {
      clearInterval(interval);
    }
    return () => clearInterval(interval);
  }, [isPlaying, playbackSpeed, selectedRun, hasVideoError]);

  // Sync isPlaying with videoRef
  useEffect(() => {
    if (videoRef.current && !hasVideoError) {
      if (isPlaying) {
        videoRef.current.play().catch(() => {});
      } else {
        videoRef.current.pause();
      }
    }
  }, [isPlaying, hasVideoError]);

  // Execute Natural Language Temporal Query
  const handleExecuteQuery = async (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    if (!queryInput.trim() || !selectedRun) return;

    setIsQuerying(true);
    setQueryResult(null);
    try {
      const res = await fetch("/api/v1/query/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query_text: queryInput.trim(),
          question: queryInput.trim(),
          run_id: selectedRun.run_id,
        }),
      });
      if (res.ok) {
        const data = await res.json();
        setQueryResult(data);
      } else {
        const err = await res.json().catch(() => ({}));
        setQueryResult({
          status: "ERROR",
          explanation: err.detail || "Query evaluation failed.",
          summary: err.detail || "Query evaluation failed.",
          evidence: [],
          evidence_items: [],
        });
      }
    } catch (err: any) {
      setQueryResult({
        status: "ERROR",
        explanation: err.message || "Failed to execute query against Visual Query Layer.",
        summary: err.message || "Failed to execute query.",
        evidence: [],
        evidence_items: [],
      });
    } finally {
      setIsQuerying(false);
    }
  };

  // Create Region ROI
  const handleCreateRegion = async () => {
    if (!selectedRun) return;
    try {
      const res = await fetch("/api/v1/events/regions", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          video_id: selectedRun.video_id,
          name: newRegionName,
          shape_type: "RECTANGLE",
          coordinates: [
            [200, 200],
            [700, 700],
          ],
          color: newRegionColor,
        }),
      });
      if (res.ok) {
        setIsRegionModalOpen(false);
        fetchRegions(selectedRun.video_id);
        fetchEvents(selectedRun.run_id);
      }
    } catch (err) {
      console.error("Failed creating region:", err);
    }
  };

  // Compare Videos
  const handleCompareVideos = async () => {
    if (!runs || runs.length < 2) return;
    try {
      const res = await fetch("/api/v1/video/compare", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          video_a_id: runs[0].video_id,
          video_b_id: runs[1].video_id,
        }),
      });
      if (res.ok) {
        const data = await res.json();
        setCompareResult(data);
        setIsCompareModalOpen(true);
      }
    } catch (err) {
      console.error("Comparison error:", err);
    }
  };

  // Jump to specific timestamp
  const seekTo = (sec: number) => {
    const clamped = Math.max(0, Math.min(selectedRun?.duration_sec || 10, sec));
    setCurrentTimeSec(parseFloat(clamped.toFixed(2)));
    if (videoRef.current && !hasVideoError) {
      videoRef.current.currentTime = clamped;
    }
  };

  // Toggle Play / Pause
  const handleTogglePlay = () => {
    setIsPlaying((prev) => !prev);
  };

  // Change Speed
  const handleSpeedChange = (speed: number) => {
    setPlaybackSpeed(speed);
    if (videoRef.current && !hasVideoError) {
      videoRef.current.playbackRate = speed;
    }
  };

  // Active tracks at current time
  const activeTracksAtCurrentTime =
    selectedRun?.tracks.filter(
      (t) => currentTimeSec >= t.first_timestamp_sec - 0.2 && currentTimeSec <= t.last_timestamp_sec + 0.2
    ) || [];

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 p-6 space-y-6">
      {/* Top Header */}
      <PageHeader
        title="Video Understanding & Temporal Intelligence Lab"
        description="Continuous visual trajectory tracking, rule-based temporal event extraction, spatial ROI zones, and natural language query evidence."
        actions={
          <div className="flex items-center gap-3">
            <Button
              variant="outline"
              className="border-slate-700 bg-slate-900/60 hover:bg-slate-800 text-slate-200"
              onClick={() => setIsUploadModalOpen(true)}
            >
              <Upload className="w-4 h-4 mr-2 text-blue-400" />
              Upload Video
            </Button>

            <Button
              variant="outline"
              className="border-slate-700 bg-slate-900/60 hover:bg-slate-800 text-slate-200"
              onClick={handleCompareVideos}
            >
              <Split className="w-4 h-4 mr-2 text-indigo-400" />
              Compare Runs
            </Button>

            <Button
              variant="outline"
              className="border-slate-700 bg-slate-900/60 hover:bg-slate-800 text-slate-200"
              onClick={() => setIsRegionModalOpen(true)}
            >
              <MapPin className="w-4 h-4 mr-2 text-emerald-400" />
              Define ROI Zone
            </Button>

            <Link href={`/api/v1/video/runs/${selectedRun?.run_id}/export`} target="_blank">
              <Button variant="outline" className="border-slate-700 bg-slate-900/60 hover:bg-slate-800 text-slate-200">
                <Download className="w-4 h-4 mr-2 text-sky-400" />
                Export CSV
              </Button>
            </Link>
          </div>
        }
      />

      {/* Main Workspace Layout */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left / Center: Interactive Video Canvas & Controls (7 Cols) */}
        <div className="lg:col-span-7 space-y-4">
          {/* Video Player Canvas Card */}
          <div className="bg-slate-900/70 border border-slate-800 rounded-xl overflow-hidden shadow-2xl backdrop-blur-sm">
            <div className="p-4 border-b border-slate-800/80 flex flex-wrap items-center justify-between gap-3 bg-slate-900/90">
              <div className="flex items-center gap-3">
                <div className="w-2.5 h-2.5 rounded-full bg-emerald-500 animate-pulse flex-shrink-0" />
                <span className="text-xs text-slate-400 font-mono flex-shrink-0">VIDEO:</span>
                <select
                  value={selectedRun?.run_id || ""}
                  onChange={(e) => {
                    const found = runs.find((r) => r.run_id === e.target.value);
                    if (found) {
                      setSelectedRun(found);
                      setCurrentTimeSec(0.0);
                      fetchRegions(found.video_id);
                      fetchEvents(found.run_id);
                    }
                  }}
                  className="bg-slate-950 border border-slate-700 hover:border-blue-500 rounded-lg px-3 py-1.5 text-xs font-semibold text-white focus:outline-none focus:ring-1 focus:ring-blue-500 transition-colors max-w-[260px] md:max-w-[340px] truncate"
                >
                  {runs.map((r) => {
                    const videoMeta = videos.find((v) => v.video_id === r.video_id);
                    const filename =
                      videoMeta?.filename ||
                      (r.video_id === "sample_traffic_01"
                        ? "warehouse_security_stream_01.mp4"
                        : `${r.video_id}.mp4`);
                    const isCustom = r.video_id.startsWith("vid_");
                    return (
                      <option key={r.run_id} value={r.run_id}>
                        {filename} {isCustom ? "(Custom Upload)" : "(Demo Stream)"} — {r.total_tracks} tracks ({r.duration_sec}s)
                      </option>
                    );
                  })}
                </select>

                <span className="text-xs px-2 py-1 rounded bg-slate-800 text-slate-300 font-mono hidden sm:inline-block">
                  {selectedSession?.width || 1920}x{selectedSession?.height || 1080} @ {selectedSession?.fps || 30} FPS
                </span>
              </div>
              <div className="flex items-center gap-2 text-xs text-slate-400 font-mono bg-slate-950 px-2.5 py-1 rounded border border-slate-800">
                <span>
                  {currentTimeSec.toFixed(2)}s / {(selectedRun?.duration_sec || 10.0).toFixed(2)}s
                </span>
              </div>
            </div>

            {/* Video Canvas Simulation & Native Video Player Screen */}
            <div className="relative aspect-video bg-slate-950 flex items-center justify-center overflow-hidden group select-none">
              {/* Native Video Element if video stream is available */}
              <video
                ref={videoRef}
                key={selectedRun?.video_id}
                src={`/api/v1/video/stream/${selectedRun?.video_id}`}
                className={`absolute inset-0 w-full h-full object-contain ${
                  hasVideoError ? "opacity-0 pointer-events-none" : "opacity-100"
                }`}
                muted
                playsInline
                onError={() => setHasVideoError(true)}
                onLoadedMetadata={() => setHasVideoError(false)}
                onTimeUpdate={(e) => {
                  if (!hasVideoError) {
                    setCurrentTimeSec(parseFloat(e.currentTarget.currentTime.toFixed(2)));
                  }
                }}
                onEnded={() => setIsPlaying(false)}
              />

              {/* Canvas Background Simulation (fallback when no video stream or background) */}
              {hasVideoError && (
                <>
                  <div className="absolute inset-0 bg-gradient-to-br from-slate-900/70 via-slate-950 to-slate-900/80" />
                  <div className="absolute inset-0 opacity-15 bg-[radial-gradient(#38bdf8_1px,transparent_1px)] [background-size:24px_24px]" />
                </>
              )}

              {/* Render Active ROI Zones */}
              {showRegions &&
                regions.map((reg) => {
                  const vidW = selectedSession?.width || 1920;
                  const vidH = selectedSession?.height || 1080;
                  const leftPct = (reg.coordinates[0][0] / vidW) * 100;
                  const topPct = (reg.coordinates[0][1] / vidH) * 100;
                  const widthPct = ((reg.coordinates[1][0] - reg.coordinates[0][0]) / vidW) * 100;
                  const heightPct = ((reg.coordinates[1][1] - reg.coordinates[0][1]) / vidH) * 100;

                  return (
                    <div
                      key={reg.region_id}
                      className="absolute border-2 border-dashed rounded-lg bg-indigo-500/10 pointer-events-none transition-all duration-300 z-10"
                      style={{
                        left: `${leftPct}%`,
                        top: `${topPct}%`,
                        width: `${widthPct}%`,
                        height: `${heightPct}%`,
                        borderColor: reg.color || "#3b82f6",
                      }}
                    >
                      <div
                        className="absolute top-1 left-1 text-[10px] font-medium px-1.5 py-0.5 rounded text-white shadow-md backdrop-blur-md"
                        style={{ backgroundColor: reg.color || "#3b82f6" }}
                      >
                        {reg.name}
                      </div>
                    </div>
                  );
                })}

              {/* Render Active Tracks & Bounding Boxes */}
              {activeTracksAtCurrentTime.map((track) => {
                const vidW = selectedSession?.width || 1920;
                const vidH = selectedSession?.height || 1080;

                // Find closest trajectory point for current time
                const point = track.trajectory.reduce((prev, curr) =>
                  Math.abs(curr.timestamp_sec - currentTimeSec) < Math.abs(prev.timestamp_sec - currentTimeSec)
                    ? curr
                    : prev,
                  track.trajectory[0]
                );

                if (!point) return null;

                const leftPct = (point.bbox[0] / vidW) * 100;
                const topPct = (point.bbox[1] / vidH) * 100;
                const widthPct = ((point.bbox[2] - point.bbox[0]) / vidW) * 100;
                const heightPct = ((point.bbox[3] - point.bbox[1]) / vidH) * 100;

                const isSelected = selectedTrack?.track_id === track.track_id;

                return (
                  <div
                    key={track.track_id}
                    onClick={() => setSelectedTrack(track)}
                    className={`absolute cursor-pointer transition-all duration-150 z-20 ${
                      showBoxes ? "border-2 rounded" : ""
                    } ${
                      isSelected
                        ? "border-amber-400 bg-amber-400/20 ring-2 ring-amber-400/50"
                        : "border-sky-400 bg-sky-400/10 hover:border-sky-300"
                    }`}
                    style={{
                      left: `${leftPct}%`,
                      top: `${topPct}%`,
                      width: `${widthPct}%`,
                      height: `${heightPct}%`,
                    }}
                  >
                    {showTrackIds && (
                      <div className="absolute -top-6 left-0 bg-slate-900/90 text-sky-300 text-[11px] font-mono font-bold px-1.5 py-0.5 rounded shadow-lg border border-sky-500/40 flex items-center gap-1 whitespace-nowrap">
                        <span>#{track.track_id}</span>
                        <span className="text-slate-400 font-normal">({track.class_name})</span>
                        <span className="text-emerald-400">{(track.avg_confidence * 100).toFixed(0)}%</span>
                      </div>
                    )}
                  </div>
                );
              })}

              {/* Render Motion Trails / Trajectories */}
              {showTrajectories && selectedTrack && (
                <svg className="absolute inset-0 w-full h-full pointer-events-none z-15">
                  <polyline
                    fill="none"
                    stroke="#f59e0b"
                    strokeWidth="3"
                    strokeDasharray="4 2"
                    points={selectedTrack.trajectory
                      .map((pt) => {
                        const vidW = selectedSession?.width || 1920;
                        const vidH = selectedSession?.height || 1080;
                        return `${(pt.x_center_px / vidW) * 100}%,${(pt.y_center_px / vidH) * 100}%`;
                      })
                      .join(" ")}
                  />
                  {selectedTrack.trajectory.map((pt, i) => {
                    const vidW = selectedSession?.width || 1920;
                    const vidH = selectedSession?.height || 1080;
                    return (
                      <circle
                        key={i}
                        cx={`${(pt.x_center_px / vidW) * 100}%`}
                        cy={`${(pt.y_center_px / vidH) * 100}%`}
                        r="3"
                        fill="#fbbf24"
                      />
                    );
                  })}
                </svg>
              )}

              {/* Empty state if video not active */}
              {activeTracksAtCurrentTime.length === 0 && (
                <div className="text-center text-slate-500 pointer-events-none z-0">
                  <Activity className="w-8 h-8 mx-auto mb-1 text-slate-600 animate-pulse" />
                  <p className="text-xs">No active tracks at t={currentTimeSec.toFixed(2)}s</p>
                </div>
              )}
            </div>

            {/* Playback Controls & Timeline Scrubber */}
            <div className="p-4 bg-slate-900/90 border-t border-slate-800 space-y-3">
              {/* Scrubber Bar */}
              <div className="space-y-1">
                <input
                  type="range"
                  min="0"
                  max={selectedRun?.duration_sec || 10.0}
                  step="0.05"
                  value={currentTimeSec}
                  onChange={(e) => seekTo(parseFloat(e.target.value))}
                  className="w-full h-2 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-sky-500"
                />
                <div className="flex justify-between text-[11px] text-slate-500 font-mono">
                  <span>0.00s</span>
                  <span>{((selectedRun?.duration_sec || 10.0) / 2).toFixed(2)}s</span>
                  <span>{(selectedRun?.duration_sec || 10.0).toFixed(2)}s</span>
                </div>
              </div>

              {/* Control Buttons */}
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Button
                    variant="outline"
                    size="sm"
                    className="border-slate-700 bg-slate-800 hover:bg-slate-700 text-slate-200 w-9 h-9 p-0"
                    onClick={handleTogglePlay}
                  >
                    {isPlaying ? <Pause className="w-4 h-4 text-amber-400" /> : <Play className="w-4 h-4 text-emerald-400" />}
                  </Button>

                  <Button
                    variant="ghost"
                    size="sm"
                    className="text-slate-400 hover:text-slate-200 text-xs font-mono"
                    onClick={() => seekTo(currentTimeSec - 1.0)}
                  >
                    -1s
                  </Button>

                  <Button
                    variant="ghost"
                    size="sm"
                    className="text-slate-400 hover:text-slate-200 text-xs font-mono"
                    onClick={() => seekTo(currentTimeSec + 1.0)}
                  >
                    +1s
                  </Button>

                  <Button
                    variant="ghost"
                    size="sm"
                    className="text-slate-400 hover:text-slate-200 text-xs font-mono"
                    onClick={() => seekTo(0)}
                  >
                    <RotateCcw className="w-3.5 h-3.5" />
                  </Button>
                </div>

                {/* Speed selector */}
                <div className="flex items-center gap-1 bg-slate-950 p-1 rounded-lg border border-slate-800">
                  {[0.5, 1.0, 2.0].map((spd) => (
                    <button
                      key={spd}
                      onClick={() => handleSpeedChange(spd)}
                      className={`text-xs px-2 py-0.5 rounded font-mono transition-colors ${
                        playbackSpeed === spd ? "bg-sky-600 text-white font-bold" : "text-slate-400 hover:text-slate-200"
                      }`}
                    >
                      {spd}x
                    </button>
                  ))}
                </div>

                {/* Overlay Toggle Buttons */}
                <div className="flex items-center gap-1.5">
                  <button
                    onClick={() => setShowBoxes(!showBoxes)}
                    className={`p-1.5 rounded text-xs flex items-center gap-1 border ${
                      showBoxes ? "bg-sky-500/20 border-sky-500 text-sky-300" : "bg-slate-950 border-slate-800 text-slate-500"
                    }`}
                    title="Toggle Bounding Boxes"
                  >
                    <Box className="w-3.5 h-3.5" />
                    <span className="hidden sm:inline">Boxes</span>
                  </button>

                  <button
                    onClick={() => setShowTrajectories(!showTrajectories)}
                    className={`p-1.5 rounded text-xs flex items-center gap-1 border ${
                      showTrajectories
                        ? "bg-amber-500/20 border-amber-500 text-amber-300"
                        : "bg-slate-950 border-slate-800 text-slate-500"
                    }`}
                    title="Toggle Trajectories"
                  >
                    <TrendingUp className="w-3.5 h-3.5" />
                    <span className="hidden sm:inline">Trails</span>
                  </button>

                  <button
                    onClick={() => setShowRegions(!showRegions)}
                    className={`p-1.5 rounded text-xs flex items-center gap-1 border ${
                      showRegions ? "bg-emerald-500/20 border-emerald-500 text-emerald-300" : "bg-slate-950 border-slate-800 text-slate-500"
                    }`}
                    title="Toggle ROI Regions"
                  >
                    <MapPin className="w-3.5 h-3.5" />
                    <span className="hidden sm:inline">Zones</span>
                  </button>
                </div>
              </div>
            </div>
          </div>

          {/* Ask VisionForge Temporal Query Bar */}
          <div className="bg-slate-900/70 border border-slate-800 rounded-xl p-4 shadow-xl backdrop-blur-sm">
            <form onSubmit={handleExecuteQuery} className="flex gap-2">
              <div className="relative flex-1">
                <Search className="w-4 h-4 absolute left-3 top-3 text-slate-400" />
                <input
                  type="text"
                  placeholder="Ask VisionForge (e.g. 'What objects entered Zone A?', 'Which person stayed longest?', 'How many cars?')..."
                  value={queryInput}
                  onChange={(e) => setQueryInput(e.target.value)}
                  className="w-full pl-9 pr-4 py-2 bg-slate-950 border border-slate-800 rounded-lg text-sm text-slate-200 placeholder:text-slate-500 focus:outline-none focus:border-sky-500"
                />
              </div>
              <Button type="submit" disabled={isQuerying} className="bg-sky-600 hover:bg-sky-500 text-white text-xs px-4">
                {isQuerying ? <RefreshCw className="w-4 h-4 animate-spin mr-1" /> : <Sparkle className="w-4 h-4 mr-1" />}
                Query
              </Button>
            </form>

            {/* Query Results Preview */}
            {queryResult && (
              <div className="mt-3 p-3 rounded-lg bg-slate-950 border border-slate-800/80 space-y-2">
                <div className="flex items-center justify-between text-xs">
                  <span className="text-slate-400 font-mono">
                    DSL: {queryResult.structured_query?.query_type || queryResult.result_type || "QUERY"}
                  </span>
                  <span
                    className={`px-2 py-0.5 rounded text-[10px] font-bold ${
                      queryResult.status === "SUCCESS" ? "bg-emerald-500/20 text-emerald-300" : "bg-amber-500/20 text-amber-300"
                    }`}
                  >
                    {queryResult.status}
                  </span>
                </div>
                <p className="text-xs text-slate-200 font-medium">
                  {queryResult.summary || queryResult.explanation || queryResult.interpretation_explanation}
                </p>

                {/* Evidence timeline items with jump-to-time buttons */}
                {(queryResult.evidence || queryResult.evidence_items || []).length > 0 && (
                  <div className="flex flex-wrap gap-2 pt-1">
                    {(queryResult.evidence || queryResult.evidence_items).map((item: any, i: number) => {
                      const tSec = item.timestamp_sec ?? (item.frame_index ? item.frame_index / 30.0 : 0.0);
                      return (
                        <button
                          key={i}
                          type="button"
                          onClick={() => seekTo(tSec)}
                          className="text-[11px] px-2 py-1 bg-slate-900 border border-slate-800 hover:border-sky-500 rounded text-sky-400 font-mono flex items-center gap-1 transition-colors"
                        >
                          <Clock className="w-3 h-3" />
                          <span>{item.description || item.event_type || `Evidence #${i + 1}`} (t={tSec.toFixed(1)}s)</span>
                        </button>
                      );
                    })}
                  </div>
                )}
              </div>
            )}
          </div>
        </div>

        {/* Right: Tabbed Intelligence Panel (5 Cols) */}
        <div className="lg:col-span-5 space-y-4">
          {/* Navigation Tabs */}
          <div className="flex border-b border-slate-800 bg-slate-900/80 rounded-t-xl p-1 gap-1">
            {[
              { id: "timeline", label: "Event Stream", icon: History },
              { id: "tracks", label: "Tracks & Replay", icon: Activity },
              { id: "regions", label: "ROI Zones", icon: MapPin },
              { id: "lineage", label: "Lineage", icon: ShieldCheck },
            ].map((tab) => {
              const Icon = tab.icon;
              return (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id as any)}
                  className={`flex-1 py-2 text-xs font-medium rounded-lg flex items-center justify-center gap-1.5 transition-colors ${
                    activeTab === tab.id
                      ? "bg-slate-800 text-sky-400 font-semibold shadow"
                      : "text-slate-400 hover:text-slate-200"
                  }`}
                >
                  <Icon className="w-3.5 h-3.5" />
                  <span>{tab.label}</span>
                </button>
              );
            })}
          </div>

          {/* Tab 1: Event Stream */}
          {activeTab === "timeline" && (
            <div className="bg-slate-900/70 border border-slate-800 rounded-b-xl p-4 space-y-3 max-h-[600px] overflow-y-auto">
              <div className="flex items-center justify-between text-xs text-slate-400 pb-2 border-b border-slate-800">
                <span>Detected Observable Events ({events.length})</span>
                <span className="text-[10px] text-slate-500 font-mono">Sorted Chronologically</span>
              </div>

              {events.map((evt) => (
                <div
                  key={evt.event_id}
                  onClick={() => {
                    setSelectedEvent(evt);
                    seekTo(evt.start_timestamp_sec);
                  }}
                  className={`p-3 rounded-lg border transition-all cursor-pointer ${
                    selectedEvent?.event_id === evt.event_id
                      ? "bg-sky-950/40 border-sky-500 ring-1 ring-sky-500/40"
                      : "bg-slate-950 border-slate-800/80 hover:border-slate-700"
                  }`}
                >
                  <div className="flex items-center justify-between mb-1">
                    <span className="text-[10px] font-mono font-bold px-1.5 py-0.5 rounded bg-slate-900 text-sky-400 border border-slate-800">
                      {evt.event_type}
                    </span>
                    <span className="text-[11px] font-mono text-slate-400 flex items-center gap-1">
                      <Clock className="w-3 h-3 text-slate-500" />
                      t={evt.start_timestamp_sec.toFixed(1)}s
                      {evt.duration_sec > 0 && ` (${evt.duration_sec.toFixed(1)}s)`}
                    </span>
                  </div>

                  <p className="text-xs text-slate-200 mb-2">{evt.description}</p>

                  {/* Trigger Rule Pill */}
                  {evt.trigger_rule && (
                    <div className="text-[11px] text-slate-400 bg-slate-900/80 p-1.5 rounded border border-slate-800 font-mono">
                      <span className="text-amber-400 font-semibold">Trigger Basis: </span>
                      {evt.trigger_rule}
                    </div>
                  )}

                  {/* Deep link actions */}
                  <div className="mt-2 pt-2 border-t border-slate-800/60 flex items-center justify-between text-[11px]">
                    <span className="text-slate-500 font-mono">Tracks: #{evt.source_track_ids.join(", #")}</span>
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        seekTo(evt.start_timestamp_sec);
                      }}
                      className="text-sky-400 hover:text-sky-300 font-medium flex items-center gap-1"
                    >
                      Jump to Frame <ArrowRight className="w-3 h-3" />
                    </button>
                  </div>
                </div>
              ))}
            </div>
          )}

          {/* Tab 2: Track Inspector & Replay */}
          {activeTab === "tracks" && (
            <div className="bg-slate-900/70 border border-slate-800 rounded-b-xl p-4 space-y-3 max-h-[600px] overflow-y-auto">
              <div className="text-xs text-slate-400 pb-2 border-b border-slate-800">
                Tracked Object Lifecycles ({selectedRun?.tracks.length || 0})
              </div>

              {selectedRun?.tracks.map((track) => {
                const isSelected = selectedTrack?.track_id === track.track_id;
                return (
                  <div
                    key={track.track_id}
                    onClick={() => {
                      setSelectedTrack(track);
                      seekTo(track.first_timestamp_sec);
                    }}
                    className={`p-3 rounded-lg border transition-all cursor-pointer ${
                      isSelected
                        ? "bg-amber-950/30 border-amber-500 ring-1 ring-amber-500/40"
                        : "bg-slate-950 border-slate-800/80 hover:border-slate-700"
                    }`}
                  >
                    <div className="flex items-center justify-between mb-2">
                      <div className="flex items-center gap-2">
                        <span className="font-mono font-bold text-xs text-amber-400">Track #{track.track_id}</span>
                        <span className="text-xs px-2 py-0.5 bg-slate-900 rounded text-slate-300 border border-slate-800">
                          {track.class_name}
                        </span>
                      </div>
                      <span className="text-xs text-emerald-400 font-mono">
                        {(track.avg_confidence * 100).toFixed(0)}% conf
                      </span>
                    </div>

                    {/* Measurable Telemetry Grid */}
                    <div className="grid grid-cols-2 gap-2 text-[11px] font-mono text-slate-400 bg-slate-900/60 p-2 rounded border border-slate-800/60">
                      <div>
                        <span className="text-slate-500">Duration: </span>
                        {track.visibility_duration_sec.toFixed(1)}s
                      </div>
                      <div>
                        <span className="text-slate-500">Displacement: </span>
                        {track.total_distance_px.toFixed(0)}px
                      </div>
                      <div>
                        <span className="text-slate-500">Velocity: </span>
                        {track.image_space_velocity_px_s.toFixed(1)} px/s
                      </div>
                      <div>
                        <span className="text-slate-500">Observations: </span>
                        {track.observation_count || track.detections_count} frames
                      </div>
                    </div>

                    {/* Visited Regions */}
                    {track.regions_visited && track.regions_visited.length > 0 && (
                      <div className="mt-2 text-[11px] text-slate-400">
                        <span className="text-slate-500">Zones Entered: </span>
                        {track.regions_visited.map((rv, i) => (
                          <span key={i} className="inline-block px-1.5 py-0.5 bg-slate-900 text-sky-300 rounded mr-1">
                            {rv.region_name} ({rv.dwell_duration_sec.toFixed(1)}s)
                          </span>
                        ))}
                      </div>
                    )}

                    {/* Cross-System Deep Links */}
                    <div className="mt-3 pt-2 border-t border-slate-800 flex items-center justify-between text-[11px]">
                      <Link
                        href={`/visual-search?query=track_${track.track_id}`}
                        className="text-slate-400 hover:text-sky-400"
                        onClick={(e) => e.stopPropagation()}
                      >
                        [Find Similar]
                      </Link>
                      <Link
                        href={`/explainability?track_id=${track.track_id}`}
                        className="text-slate-400 hover:text-amber-400"
                        onClick={(e) => e.stopPropagation()}
                      >
                        [Explain]
                      </Link>
                      <Link
                        href={`/evaluation?focus_track=${track.track_id}`}
                        className="text-slate-400 hover:text-emerald-400"
                        onClick={(e) => e.stopPropagation()}
                      >
                        [View Failure]
                      </Link>
                    </div>
                  </div>
                );
              })}
            </div>
          )}

          {/* Tab 3: Regions of Interest (ROI) */}
          {activeTab === "regions" && (
            <div className="bg-slate-900/70 border border-slate-800 rounded-b-xl p-4 space-y-3 max-h-[600px] overflow-y-auto">
              <div className="flex items-center justify-between text-xs text-slate-400 pb-2 border-b border-slate-800">
                <span>Configured Spatial ROIs ({regions.length})</span>
                <Button
                  size="sm"
                  variant="ghost"
                  className="text-sky-400 hover:text-sky-300 text-xs p-0 h-auto"
                  onClick={() => setIsRegionModalOpen(true)}
                >
                  + Add Zone
                </Button>
              </div>

              {regions.map((reg) => (
                <div key={reg.region_id} className="p-3 rounded-lg bg-slate-950 border border-slate-800 space-y-2">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <div className="w-3 h-3 rounded-full" style={{ backgroundColor: reg.color || "#3b82f6" }} />
                      <span className="text-xs font-semibold text-slate-200">{reg.name}</span>
                    </div>
                    <span className="text-[10px] font-mono text-slate-500">{reg.shape_type}</span>
                  </div>

                  <div className="text-[11px] font-mono text-slate-400 bg-slate-900/80 p-2 rounded border border-slate-800">
                    <div>Coordinates: [{reg.coordinates.map((c) => `[${c.join(",")}]`).join(", ")}]</div>
                    <div>Reference: {reg.coordinate_system} Coordinate Space</div>
                  </div>
                </div>
              ))}
            </div>
          )}

          {/* Tab 4: Session Lineage & Reproducibility */}
          {activeTab === "lineage" && (
            <div className="bg-slate-900/70 border border-slate-800 rounded-b-xl p-4 space-y-3 max-h-[600px] overflow-y-auto">
              <div className="text-xs text-slate-400 pb-2 border-b border-slate-800">
                Lineage & Execution Provenance
              </div>

              <div className="bg-slate-950 p-3 rounded-lg border border-slate-800 space-y-2 text-xs font-mono">
                <div className="flex justify-between">
                  <span className="text-slate-500">Session ID:</span>
                  <span className="text-slate-300">{selectedSession?.session_id || "vses_default"}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-500">Model Checkpoint:</span>
                  <span className="text-sky-400">{selectedRun?.model_id || "yolo11s.pt"}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-500">Tracker Algorithm:</span>
                  <span className="text-amber-400">{selectedRun?.tracker_name || "ByteTrack (IoU)"}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-500">Processing FPS:</span>
                  <span className="text-emerald-400">{selectedRun?.processing_fps || 30.0} FPS</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-500">Inference Latency:</span>
                  <span className="text-slate-300">{selectedRun?.inference_latency_ms || 12.5} ms</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-500">Cryptographic Hash:</span>
                  <span className="text-slate-400 text-[10px]">{selectedSession?.video_fingerprint || "sha256_verified"}</span>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Define ROI Modal */}
      {isRegionModalOpen && (
        <div className="fixed inset-0 z-50 bg-slate-950/80 backdrop-blur-sm flex items-center justify-center p-4">
          <div className="bg-slate-900 border border-slate-800 rounded-xl max-w-md w-full p-5 space-y-4 shadow-2xl">
            <h3 className="text-base font-semibold text-slate-100 flex items-center gap-2">
              <MapPin className="w-5 h-5 text-emerald-400" />
              Define Region of Interest (ROI)
            </h3>

            <div className="space-y-3 text-xs">
              <div>
                <label className="block text-slate-400 mb-1">Region Name</label>
                <input
                  type="text"
                  value={newRegionName}
                  onChange={(e) => setNewRegionName(e.target.value)}
                  className="w-full px-3 py-2 bg-slate-950 border border-slate-800 rounded text-slate-200"
                />
              </div>

              <div>
                <label className="block text-slate-400 mb-1">Stroke Color</label>
                <input
                  type="color"
                  value={newRegionColor}
                  onChange={(e) => setNewRegionColor(e.target.value)}
                  className="w-full h-8 bg-slate-950 border border-slate-800 rounded cursor-pointer"
                />
              </div>
            </div>

            <div className="flex justify-end gap-2 pt-2 border-t border-slate-800">
              <Button variant="ghost" size="sm" onClick={() => setIsRegionModalOpen(false)}>
                Cancel
              </Button>
              <Button size="sm" className="bg-emerald-600 hover:bg-emerald-500 text-white" onClick={handleCreateRegion}>
                Save Region
              </Button>
            </div>
          </div>
        </div>
      )}

      {/* Video Comparison Modal */}
      {isCompareModalOpen && compareResult && (
        <div className="fixed inset-0 z-50 bg-slate-950/80 backdrop-blur-sm flex items-center justify-center p-4">
          <div className="bg-slate-900 border border-slate-800 rounded-xl max-w-lg w-full p-5 space-y-4 shadow-2xl">
            <h3 className="text-base font-semibold text-slate-100 flex items-center gap-2">
              <Split className="w-5 h-5 text-indigo-400" />
              Side-by-Side Video Intelligence Comparison
            </h3>

            <div className="space-y-2 text-xs font-mono bg-slate-950 p-3 rounded-lg border border-slate-800">
              <div className="flex justify-between">
                <span className="text-slate-500">Track Count Delta:</span>
                <span className={compareResult.track_count_delta >= 0 ? "text-emerald-400" : "text-amber-400"}>
                  {compareResult.track_count_delta > 0 ? `+${compareResult.track_count_delta}` : compareResult.track_count_delta} tracks
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-500">Average Dwell Delta:</span>
                <span className="text-slate-200">{compareResult.avg_dwell_delta_sec}s</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-500">Event Volume Delta:</span>
                <span className="text-slate-200">{compareResult.event_count_delta} observations</span>
              </div>
            </div>

            <div className="space-y-1">
              <span className="text-xs font-semibold text-slate-300">Summary Findings:</span>
              <ul className="text-xs text-slate-400 list-disc list-inside space-y-1">
                {compareResult.summary_findings.map((f: string, i: number) => (
                  <li key={i}>{f}</li>
                ))}
              </ul>
            </div>

            <div className="flex justify-end pt-2 border-t border-slate-800">
              <Button size="sm" variant="ghost" onClick={() => setIsCompareModalOpen(false)}>
                Close
              </Button>
            </div>
          </div>
        </div>
      )}

      {/* Upload Custom Video Modal */}
      {isUploadModalOpen && (
        <div className="fixed inset-0 z-50 bg-slate-950/80 backdrop-blur-sm flex items-center justify-center p-4">
          <div className="bg-slate-900 border border-slate-800 rounded-xl max-w-md w-full p-5 space-y-4 shadow-2xl">
            <h3 className="text-base font-semibold text-slate-100 flex items-center gap-2">
              <Upload className="w-5 h-5 text-blue-400" />
              Upload Custom Video Asset
            </h3>

            <p className="text-xs text-slate-400">
              Upload any MP4, MOV, AVI, or MKV video. VisionForge will automatically parse metadata and execute object detection & ByteTrack tracking.
            </p>

            <label className="relative block border-2 border-dashed border-slate-700 hover:border-blue-500/60 rounded-xl p-6 text-center transition-colors cursor-pointer bg-slate-950/50">
              <Video className="w-8 h-8 text-slate-500 mx-auto mb-2" />
              <p className="text-xs text-slate-300 font-medium">
                {uploadFile ? (
                  <span className="text-emerald-400 font-mono">{uploadFile.name}</span>
                ) : (
                  <>Drag & Drop or <span className="text-blue-400">Browse Video</span></>
                )}
              </p>
              <p className="text-[10px] text-slate-500 mt-1">MP4, MOV, AVI, MKV up to 500MB</p>
              <input
                type="file"
                accept="video/*"
                onChange={(e) => {
                  if (e.target.files && e.target.files[0]) {
                    setUploadFile(e.target.files[0]);
                  }
                }}
                className="hidden"
              />
            </label>

            {uploadError && (
              <div className="text-xs text-rose-400 bg-rose-950/30 border border-rose-800/40 p-2.5 rounded-lg flex items-center gap-2">
                <AlertCircle className="w-4 h-4 flex-shrink-0" />
                <span>{uploadError}</span>
              </div>
            )}

            <div className="flex justify-end gap-2 pt-2 border-t border-slate-800">
              <Button
                variant="ghost"
                size="sm"
                disabled={isUploading}
                onClick={() => {
                  setIsUploadModalOpen(false);
                  setUploadFile(null);
                  setUploadError(null);
                }}
              >
                Cancel
              </Button>
              <Button
                size="sm"
                disabled={!uploadFile || isUploading}
                className="bg-blue-600 hover:bg-blue-500 text-white flex items-center gap-2"
                onClick={handleUploadVideo}
              >
                {isUploading ? (
                  <>
                    <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                    Processing Video...
                  </>
                ) : (
                  <>
                    <Upload className="w-3.5 h-3.5" />
                    Upload & Track
                  </>
                )}
              </Button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
