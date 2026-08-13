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
  Layers,
  MapPin,
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
  const [classFilter, setClassFilter] = useState<string>("ALL");
  const [loading, setLoading] = useState<boolean>(false);

  // Temporal Event Intelligence State
  const [regions, setRegions] = useState<RegionOfInterest[]>([]);
  const [events, setEvents] = useState<TemporalEvent[]>([]);
  const [selectedEventId, setSelectedEventId] = useState<string | null>(null);
  const [inspectEvidenceModal, setInspectEvidenceModal] = useState<EventEvidence | null>(null);
  const [eventFilterType, setEventFilterType] = useState<string>("ALL");
  const [showAddRegionModal, setShowAddRegionModal] = useState<boolean>(false);
  const [newRegionName, setNewRegionName] = useState<string>("Loading Zone A");

  // Event Rules Configuration State
  const [dwellThreshold, setDwellThreshold] = useState<number>(3.0);
  const [proximityThreshold, setProximityThreshold] = useState<number>(100.0);
  const [generatingEvents, setGeneratingEvents] = useState<boolean>(false);

  useEffect(() => {
    // Run initial video inference pipeline on mount
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

        // Fetch Regions & Generate Events
        await fetchRegions(data.video_id);
        await handleGenerateEvents(data.run_id);
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
    setGeneratingEvents(true);
    try {
      const res = await fetch("/api/v1/events/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          run_id: runId,
          config: {
            dwell_threshold_sec: dwellThreshold,
            proximity_threshold_px: proximityThreshold,
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
    } finally {
      setGeneratingEvents(false);
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

  // Filter active tracks at current timestamp
  const activeTracksAtCurrentTime = currentRun?.tracks.filter(
    (t) =>
      currentTimeSec >= t.first_timestamp_sec && currentTimeSec <= t.last_timestamp_sec + 0.5
  ) || [];

  // Filter events stream
  const filteredEvents = events.filter((e) => {
    if (eventFilterType !== "ALL" && e.event_type !== eventFilterType) return false;
    return true;
  });

  return (
    <div className="flex flex-col min-h-screen bg-[#0a0a0a] text-neutral-200 font-inter">
      {/* Page Header */}
      <PageHeader
        title="Video Lab & Temporal Event Intelligence"
        description="Transform low-level tracks into explainable temporal events: Region Intersections, Dwell Intervals, Proximity Events & Chronological Timeline"
        breadcrumbs={["VisionForge", "Video Lab"]}
        actions={
          <div className="flex items-center gap-2">
            <a
              href={`data:text/csv;charset=utf-8,${encodeURIComponent(
                currentRun ? `event_id,event_type,start_sec\n${events.map((e) => `${e.event_id},${e.event_type},${e.start_timestamp_sec}`).join("\n")}` : ""
              )}`}
              download={`temporal_events_${currentRun?.run_id || "export"}.csv`}
            >
              <Button variant="secondary" icon={<Download className="w-4 h-4 text-emerald-400" />}>
                Export Event CSV
              </Button>
            </a>

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
        {/* Controls Toolbar */}
        <div className="bg-[#121212] border border-white/10 rounded-xl p-4 flex flex-wrap items-center justify-between gap-4">
          <div className="flex flex-wrap items-center gap-4 text-xs">
            {/* Video Asset Dropdown */}
            <div className="flex items-center gap-2">
              <Video className="w-4 h-4 text-cyan-400" />
              <span className="text-neutral-400 font-medium">Video Asset:</span>
              <select
                value={videoId}
                onChange={(e) => setVideoId(e.target.value)}
                className="bg-[#1a1a1a] border border-white/10 rounded-lg px-3 py-1.5 text-white font-mono"
              >
                <option value="sample_traffic_01">sample_traffic_01.mp4 (10.0s, 30 FPS)</option>
                <option value="factory_safety_02">factory_safety_02.mp4 (15.0s, 30 FPS)</option>
                <option value="drone_surveillance_03">drone_surveillance_03.mp4 (20.0s, 30 FPS)</option>
              </select>
            </div>

            {/* Model Dropdown */}
            <div className="flex items-center gap-2">
              <Cpu className="w-4 h-4 text-indigo-400" />
              <span className="text-neutral-400 font-medium">Model:</span>
              <select
                value={modelId}
                onChange={(e) => setModelId(e.target.value)}
                className="bg-[#1a1a1a] border border-white/10 rounded-lg px-3 py-1.5 text-white font-mono"
              >
                <option value="yolo11s.pt">YOLO11s Safety Baseline</option>
                <option value="rtdetr_l.pt">RT-DETR-L Safety Transformer</option>
              </select>
            </div>

            {/* Tracker Dropdown */}
            <div className="flex items-center gap-2">
              <Compass className="w-4 h-4 text-amber-400" />
              <span className="text-neutral-400 font-medium">Tracker:</span>
              <select
                value={trackerName}
                onChange={(e) => setTrackerName(e.target.value)}
                className="bg-[#1a1a1a] border border-white/10 rounded-lg px-3 py-1.5 text-white font-mono"
              >
                <option value="ByteTrack">ByteTrack (IoU + Kalman)</option>
              </select>
            </div>
          </div>

          <div className="flex items-center gap-2">
            <Button
              variant="secondary"
              size="sm"
              icon={<Plus className="w-3.5 h-3.5 text-blue-400" />}
              onClick={() => setShowAddRegionModal(true)}
            >
              Add Region ROI
            </Button>

            <Button
              variant="secondary"
              size="sm"
              icon={<RefreshCw className={`w-3.5 h-3.5 ${generatingEvents ? "animate-spin" : ""}`} />}
              onClick={() => currentRun && handleGenerateEvents(currentRun.run_id)}
              disabled={generatingEvents || !currentRun}
            >
              Re-Detect Events
            </Button>
          </div>
        </div>

        {/* Main Workspace Layout */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Left Column: Video Player & Region Overlay */}
          <div className="lg:col-span-2 space-y-6">
            {/* Player Container */}
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

              {/* Video Overlay Canvas Screen */}
              <div className="relative aspect-video bg-[#080808] border border-white/10 rounded-lg overflow-hidden flex flex-col items-center justify-center">
                {/* Background Grid Pattern */}
                <div className="absolute inset-0 bg-[radial-gradient(#1f1f1f_1px,transparent_1px)] [background-size:16px_16px] opacity-40" />

                {/* SVG Layer: Regions & Trajectories */}
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

                {/* Player Center Status Telemetry */}
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

            {/* Selected Event Inspector & Evidence Actions */}
            {selectedEvent ? (
              <div className="bg-[#121212] border border-white/10 rounded-xl p-5 space-y-4">
                <div className="flex flex-wrap justify-between items-center border-b border-white/10 pb-3 gap-2">
                  <div className="flex items-center gap-3">
                    <span className="text-xs font-mono font-bold px-2 py-0.5 rounded bg-blue-600/20 text-blue-400 border border-blue-500/30">
                      {selectedEvent.event_type}
                    </span>
                    <span className="text-xs font-mono text-neutral-400">
                      ID: {selectedEvent.event_id}
                    </span>
                    <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-emerald-500/20 text-emerald-400 border border-emerald-500/30 font-bold">
                      Reliability: {selectedEvent.reliability}
                    </span>
                  </div>

                  <Button
                    variant="primary"
                    size="sm"
                    icon={<Eye className="w-3.5 h-3.5" />}
                    onClick={() => handleInspectEvidence(selectedEvent.event_id)}
                  >
                    Inspect Evidence
                  </Button>
                </div>

                <div className="text-xs text-neutral-200 font-mono bg-[#161616] p-3 rounded-lg border border-white/5 leading-relaxed">
                  {selectedEvent.description}
                </div>

                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 font-mono text-xs">
                  <div className="bg-[#181818] p-3 rounded-lg border border-white/5 space-y-1">
                    <div className="text-neutral-400 text-[10px]">Start Timestamp</div>
                    <div className="text-sm font-bold text-white">
                      t = {selectedEvent.start_timestamp_sec.toFixed(1)}s
                    </div>
                  </div>

                  <div className="bg-[#181818] p-3 rounded-lg border border-white/5 space-y-1">
                    <div className="text-neutral-400 text-[10px]">Duration</div>
                    <div className="text-sm font-bold text-blue-400">
                      {selectedEvent.duration_sec.toFixed(1)}s
                    </div>
                  </div>

                  <div className="bg-[#181818] p-3 rounded-lg border border-white/5 space-y-1">
                    <div className="text-neutral-400 text-[10px]">Source Tracks</div>
                    <div className="text-sm font-bold text-purple-400">
                      {selectedEvent.source_track_ids.map((id) => `#${id}`).join(", ") || "N/A"}
                    </div>
                  </div>

                  <div className="bg-[#181818] p-3 rounded-lg border border-white/5 space-y-1">
                    <div className="text-neutral-400 text-[10px]">Region ROI</div>
                    <div className="text-sm font-bold text-amber-400">
                      {selectedEvent.event_params.region_name || "N/A"}
                    </div>
                  </div>
                </div>
              </div>
            ) : (
              <div className="p-8 text-center text-xs text-neutral-500 bg-[#121212] border border-white/10 rounded-xl font-mono">
                Click any event in the timeline to inspect evidence and seek player timestamp.
              </div>
            )}
          </div>

          {/* Right Column: Chronological Event Timeline Stream */}
          <div className="space-y-6">
            {/* Event Timeline Stream */}
            <div className="bg-[#121212] border border-white/10 rounded-xl overflow-hidden flex flex-col h-[560px]">
              <div className="p-4 border-b border-white/10 flex justify-between items-center bg-[#161616] shrink-0">
                <h3 className="text-xs font-semibold uppercase tracking-wider text-neutral-400 flex items-center gap-2">
                  <Clock className="w-4 h-4 text-emerald-400" />
                  Chronological Event Stream ({filteredEvents.length})
                </h3>

                <select
                  value={eventFilterType}
                  onChange={(e) => setEventFilterType(e.target.value)}
                  className="bg-[#1a1a1a] border border-white/10 rounded px-2 py-1 text-[10px] text-neutral-300 font-mono"
                >
                  <option value="ALL">All Event Types</option>
                  <option value="OBJECT_ENTERED_REGION">Entered Region</option>
                  <option value="OBJECT_LEFT_REGION">Left Region</option>
                  <option value="OBJECT_DWELLED">Object Dwelled</option>
                  <option value="OBJECT_STOPPED">Object Stopped</option>
                  <option value="OBJECTS_BECAME_CLOSE">Became Close</option>
                  <option value="TRACK_STARTED">Track Started</option>
                </select>
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
                        <div className="flex items-center gap-2">
                          <span className="text-[11px] font-bold text-blue-400">
                            t={evt.start_timestamp_sec.toFixed(1)}s
                          </span>
                          <span
                            className={`text-[9px] font-bold px-1.5 py-0.5 rounded uppercase ${
                              evt.event_type.includes("DWELLED")
                                ? "bg-amber-500/20 text-amber-400 border border-amber-500/30"
                                : evt.event_type.includes("ENTERED")
                                ? "bg-emerald-500/20 text-emerald-400 border border-emerald-500/30"
                                : evt.event_type.includes("CLOSE")
                                ? "bg-purple-500/20 text-purple-400 border border-purple-500/30"
                                : "bg-neutral-800 text-neutral-300"
                            }`}
                          >
                            {evt.event_type.replace("OBJECT_", "").replace("_REGION", "")}
                          </span>
                        </div>

                        {evt.duration_sec > 0 && (
                          <span className="text-[10px] text-neutral-500">
                            {evt.duration_sec.toFixed(1)}s duration
                          </span>
                        )}
                      </div>

                      <p className="text-[11px] text-neutral-300 line-clamp-2 leading-relaxed">
                        {evt.description}
                      </p>
                    </div>
                  );
                })}

                {filteredEvents.length === 0 && (
                  <div className="py-12 text-center text-xs text-neutral-500 font-mono">
                    No temporal events match current filter.
                  </div>
                )}
              </div>
            </div>

            {/* Regions List Panel */}
            <div className="bg-[#121212] border border-white/10 rounded-xl p-4 space-y-3 font-mono text-xs">
              <div className="flex justify-between items-center border-b border-white/10 pb-2">
                <h4 className="font-semibold text-neutral-300 flex items-center gap-2">
                  <MapPin className="w-4 h-4 text-blue-400" />
                  Active Regions ROI ({regions.length})
                </h4>
                <button
                  onClick={() => setShowAddRegionModal(true)}
                  className="text-[10px] text-blue-400 hover:underline"
                >
                  + Add ROI
                </button>
              </div>

              <div className="space-y-2">
                {regions.map((reg) => (
                  <div
                    key={reg.region_id}
                    className="flex justify-between items-center bg-[#181818] p-2.5 rounded border border-white/5"
                  >
                    <div className="space-y-0.5">
                      <div className="font-bold text-white text-xs">{reg.name}</div>
                      <div className="text-[10px] text-neutral-500">{reg.shape_type} ROI</div>
                    </div>

                    <button
                      onClick={() => handleDeleteRegion(reg.region_id)}
                      className="p-1 rounded hover:bg-red-500/20 text-neutral-500 hover:text-red-400"
                      title="Delete Region"
                    >
                      <Trash2 className="w-3.5 h-3.5" />
                    </button>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Add Region Modal */}
      {showAddRegionModal && (
        <div className="fixed inset-0 bg-black/80 z-50 flex items-center justify-center p-4">
          <div className="bg-[#121212] border border-white/10 rounded-xl p-6 max-w-md w-full space-y-4 font-mono text-xs">
            <h3 className="text-sm font-semibold text-white">Create Region of Interest (ROI)</h3>
            <div>
              <label className="text-neutral-400 block mb-1">Region Name</label>
              <input
                type="text"
                value={newRegionName}
                onChange={(e) => setNewRegionName(e.target.value)}
                className="w-full bg-[#1a1a1a] border border-white/10 rounded px-3 py-2 text-white"
              />
            </div>
            <div className="text-neutral-500 text-[10px]">
              Preset Box Coordinates: [[200, 150], [1200, 700]] (Pixel space)
            </div>
            <div className="flex justify-end gap-2 pt-2">
              <Button variant="secondary" size="sm" onClick={() => setShowAddRegionModal(false)}>
                Cancel
              </Button>
              <Button variant="primary" size="sm" onClick={handleAddDefaultRegion}>
                Save Region ROI
              </Button>
            </div>
          </div>
        </div>
      )}

      {/* Inspect Evidence Modal */}
      {inspectEvidenceModal && (
        <div className="fixed inset-0 bg-black/80 z-50 flex items-center justify-center p-4">
          <div className="bg-[#121212] border border-white/10 rounded-xl p-6 max-w-2xl w-full space-y-4 font-mono text-xs">
            <div className="flex justify-between items-center border-b border-white/10 pb-3">
              <h3 className="text-sm font-semibold text-white flex items-center gap-2">
                <Eye className="w-4 h-4 text-blue-400" />
                Visual Verification Evidence: Event #{inspectEvidenceModal.event_id}
              </h3>
              <button
                onClick={() => setInspectEvidenceModal(null)}
                className="text-neutral-400 hover:text-white"
              >
                ✕
              </button>
            </div>

            <div className="grid grid-cols-3 gap-3 text-center">
              <div className="bg-[#181818] p-4 rounded-lg border border-white/5 space-y-2">
                <div className="text-[10px] text-neutral-400">Frame Before</div>
                <div className="text-xs font-bold text-white">#{inspectEvidenceModal.frame_before_idx}</div>
                <div className="h-20 bg-[#0a0a0a] rounded border border-white/5 flex items-center justify-center text-[10px] text-neutral-600">
                  Pre-Onset Frame
                </div>
              </div>

              <div className="bg-[#181818] p-4 rounded-lg border border-blue-500/40 space-y-2">
                <div className="text-[10px] text-blue-400 font-bold">Event Frame</div>
                <div className="text-xs font-bold text-blue-400">#{inspectEvidenceModal.event_frame_idx}</div>
                <div className="h-20 bg-[#0a0a0a] rounded border border-blue-500/30 flex items-center justify-center text-[10px] text-blue-400 font-bold">
                  Event Onset
                </div>
              </div>

              <div className="bg-[#181818] p-4 rounded-lg border border-white/5 space-y-2">
                <div className="text-[10px] text-neutral-400">Frame After</div>
                <div className="text-xs font-bold text-white">#{inspectEvidenceModal.frame_after_idx}</div>
                <div className="h-20 bg-[#0a0a0a] rounded border border-white/5 flex items-center justify-center text-[10px] text-neutral-600">
                  Post-Event Frame
                </div>
              </div>
            </div>

            <div className="bg-[#161616] p-3 rounded-lg border border-white/5 text-neutral-300 text-xs">
              {inspectEvidenceModal.snapshot_notes}
            </div>

            <div className="flex justify-end">
              <Button variant="secondary" size="sm" onClick={() => setInspectEvidenceModal(null)}>
                Close Evidence Inspector
              </Button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
