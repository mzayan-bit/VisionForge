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

  // Active Run & Selected Track State
  const [currentRun, setCurrentRun] = useState<VideoInferenceRun | null>(null);
  const [selectedTrackId, setSelectedTrackId] = useState<number | null>(null);
  const [showTrajectory, setShowTrajectory] = useState<boolean>(true);
  const [classFilter, setClassFilter] = useState<string>("ALL");
  const [loading, setLoading] = useState<boolean>(false);

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
      }
    } catch (err) {
      console.error("Failed to run video inference pipeline:", err);
    } finally {
      setLoading(false);
    }
  };

  const selectedTrack = currentRun?.tracks.find((t) => t.track_id === selectedTrackId);

  // Filter active tracks at current timestamp for video overlay rendering
  const activeTracksAtCurrentTime = currentRun?.tracks.filter(
    (t) =>
      currentTimeSec >= t.first_timestamp_sec && currentTimeSec <= t.last_timestamp_sec + 0.5
  ) || [];

  // Filter track gallery
  const filteredTracks = currentRun?.tracks.filter((t) => {
    if (classFilter !== "ALL" && t.class_name.toLowerCase() !== classFilter.toLowerCase())
      return false;
    return true;
  }) || [];

  const allClasses = Array.from(new Set(currentRun?.tracks.map((t) => t.class_name) || []));

  return (
    <div className="flex flex-col min-h-screen bg-[#0a0a0a] text-neutral-200 font-inter">
      {/* Page Header */}
      <PageHeader
        title="Video Lab & Multi-Object Tracking Studio"
        description="End-to-End Video Intelligence: Frame Sampling -> Object Detection -> ByteTrack Tracking -> Persistent Track IDs -> Trajectory Telemetry"
        breadcrumbs={["VisionForge", "Video Lab"]}
        actions={
          <div className="flex items-center gap-2">
            <a
              href={`data:text/csv;charset=utf-8,${encodeURIComponent(
                currentRun ? `run_id,track_id,class\n${currentRun.run_id},1,person` : ""
              )}`}
              download={`video_tracking_${currentRun?.run_id || "export"}.csv`}
            >
              <Button variant="secondary" icon={<Download className="w-4 h-4 text-emerald-400" />}>
                Export Trajectory CSV
              </Button>
            </a>

            <Button
              variant="primary"
              icon={<Video className="w-4 h-4" />}
              onClick={handleRunVideoInference}
              disabled={loading}
            >
              {loading ? "Tracking Pipeline..." : "Run Video Pipeline"}
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

            {/* Frame Sampling Stride */}
            <div className="flex items-center gap-2">
              <Sliders className="w-4 h-4 text-purple-400" />
              <span className="text-neutral-400 font-medium">Sampling:</span>
              <select
                value={samplingMode}
                onChange={(e) => setSamplingMode(e.target.value)}
                className="bg-[#1a1a1a] border border-white/10 rounded-lg px-3 py-1.5 text-white font-mono"
              >
                <option value="EVERY_FRAME">Every Frame (Stride 1)</option>
                <option value="EVERY_2ND_FRAME">Every 2nd Frame (Stride 2)</option>
                <option value="EVERY_5TH_FRAME">Every 5th Frame (Stride 5)</option>
              </select>
            </div>
          </div>

          <Button
            variant="secondary"
            size="sm"
            onClick={handleRunVideoInference}
            disabled={loading}
          >
            {loading ? "Processing..." : "Re-run Video Pipeline"}
          </Button>
        </div>

        {/* Main Workspace Layout */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Left Column: Interactive Video Player & Overlay Canvas */}
          <div className="lg:col-span-2 space-y-6">
            {/* Player Container */}
            <div className="bg-[#121212] border border-white/10 rounded-xl overflow-hidden space-y-4 p-4">
              <div className="flex justify-between items-center border-b border-white/10 pb-3">
                <h3 className="text-xs font-semibold uppercase tracking-wider text-neutral-400 flex items-center gap-2">
                  <Video className="w-4 h-4 text-cyan-400" />
                  Video Player & Multi-Object Track Overlay
                </h3>
                <span className="text-xs text-neutral-500 font-mono">
                  Timestamp: {currentTimeSec.toFixed(1)}s / {currentRun?.duration_sec.toFixed(1) || "10.0"}s
                </span>
              </div>

              {/* Video Overlay Canvas Screen */}
              <div className="relative aspect-video bg-[#080808] border border-white/10 rounded-lg overflow-hidden flex flex-col items-center justify-center">
                {/* Background Grid Pattern */}
                <div className="absolute inset-0 bg-[radial-gradient(#1f1f1f_1px,transparent_1px)] [background-size:16px_16px] opacity-40" />

                {/* SVG Bounding Boxes & Trajectories Layer */}
                <svg className="absolute inset-0 w-full h-full pointer-events-none">
                  {/* Selected Track Trajectory Path */}
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

                  {/* Active Track Overlay Bounding Boxes */}
                  {activeTracksAtCurrentTime.map((track) => {
                    const latestPt = track.trajectory.reduce(
                      (prev, curr) =>
                        curr.timestamp_sec <= currentTimeSec ? curr : prev,
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
                          fill={isSelected ? "rgba(59, 130, 246, 0.2)" : "rgba(168, 85, 247, 0.15)"}
                          stroke={isSelected ? "#3b82f6" : "#a855f7"}
                          strokeWidth={isSelected ? "3" : "1.5"}
                          rx="4"
                        />
                        {/* Track ID Badge */}
                        <foreignObject
                          x={`${latestPt.bbox[0] / 19.2}%`}
                          y={`${latestPt.bbox[1] / 10.8 - 6}%`}
                          width="120"
                          height="24"
                        >
                          <div
                            className={`text-[10px] font-mono font-bold px-1.5 py-0.5 rounded shadow w-fit ${
                              isSelected
                                ? "bg-blue-600 text-white"
                                : "bg-purple-600 text-white"
                            }`}
                          >
                            Track #{track.track_id} ({track.class_name})
                          </div>
                        </foreignObject>
                      </g>
                    );
                  })}
                </svg>

                {/* Player Center Telemetry Info */}
                <div className="z-10 text-center space-y-1">
                  <div className="text-xs font-mono text-neutral-400">
                    Simulated Video Stream ({currentRun?.processed_frames || 0} sampled frames)
                  </div>
                  <div className="text-[10px] text-neutral-600 font-mono">
                    Active Tracks: {activeTracksAtCurrentTime.length} objects visible
                  </div>
                </div>
              </div>

              {/* Playback Controls & Timeline Scrubber */}
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
                      {currentRun?.processing_fps || 0} FPS Processing
                    </span>
                  </div>
                </div>
              </div>
            </div>

            {/* Selected Track Details Inspector */}
            {selectedTrack ? (
              <div className="bg-[#121212] border border-white/10 rounded-xl p-5 space-y-4">
                <div className="flex justify-between items-center border-b border-white/10 pb-3">
                  <h4 className="text-xs font-semibold uppercase tracking-wider text-neutral-400 flex items-center gap-2">
                    <Move className="w-4 h-4 text-blue-400" />
                    Selected Track Inspector: Track #{selectedTrack.track_id} ({selectedTrack.class_name})
                  </h4>
                  <span className="text-xs font-mono text-emerald-400 font-bold">
                    Status: {selectedTrack.status}
                  </span>
                </div>

                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 font-mono text-xs">
                  <div className="bg-[#181818] p-3 rounded-lg border border-white/5 space-y-1">
                    <div className="text-neutral-400 text-[10px]">Visibility Duration</div>
                    <div className="text-sm font-bold text-white">
                      {selectedTrack.visibility_duration_sec.toFixed(1)}s
                    </div>
                  </div>

                  <div className="bg-[#181818] p-3 rounded-lg border border-white/5 space-y-1">
                    <div className="text-neutral-400 text-[10px]">Average Confidence</div>
                    <div className="text-sm font-bold text-blue-400">
                      {(selectedTrack.avg_confidence * 100).toFixed(1)}%
                    </div>
                  </div>

                  <div className="bg-[#181818] p-3 rounded-lg border border-white/5 space-y-1">
                    <div className="text-neutral-400 text-[10px]">Distance Traversed</div>
                    <div className="text-sm font-bold text-purple-400">
                      {selectedTrack.total_distance_px.toFixed(0)} px
                    </div>
                  </div>

                  <div className="bg-[#181818] p-3 rounded-lg border border-white/5 space-y-1">
                    <div className="text-neutral-400 text-[10px]">Average Pixel Speed</div>
                    <div className="text-sm font-bold text-emerald-400">
                      {selectedTrack.avg_speed_px_per_sec.toFixed(0)} px/s
                    </div>
                  </div>
                </div>

                {/* Trajectory Points Timeline */}
                <div className="space-y-2 pt-2">
                  <div className="text-[11px] font-semibold text-neutral-400 uppercase tracking-wider font-mono">
                    Spatial Trajectory History ({selectedTrack.trajectory.length} points)
                  </div>
                  <div className="bg-[#080808] border border-white/5 rounded-lg p-3 max-h-36 overflow-y-auto font-mono text-[11px] space-y-1">
                    {selectedTrack.trajectory.map((pt, idx) => (
                      <div key={idx} className="flex justify-between py-0.5 text-neutral-400 border-b border-white/5">
                        <span>t={pt.timestamp_sec.toFixed(1)}s (Frame #{pt.frame_index})</span>
                        <span className="text-blue-400">
                          Pos: ({pt.x_center_px.toFixed(0)}px, {pt.y_center_px.toFixed(0)}px)
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            ) : (
              <div className="p-8 text-center text-xs text-neutral-500 bg-[#121212] border border-white/10 rounded-xl font-mono">
                Click any track in the gallery to inspect its spatial trajectory.
              </div>
            )}
          </div>

          {/* Right Column: Track Gallery & Temporal Telemetry */}
          <div className="space-y-6">
            {/* Tracks Gallery */}
            <div className="bg-[#121212] border border-white/10 rounded-xl overflow-hidden">
              <div className="p-4 border-b border-white/10 flex justify-between items-center bg-[#161616]">
                <h3 className="text-xs font-semibold uppercase tracking-wider text-neutral-400 flex items-center gap-2">
                  <Layers className="w-4 h-4 text-purple-400" />
                  Track Gallery ({filteredTracks.length})
                </h3>
                {allClasses.length > 0 && (
                  <select
                    value={classFilter}
                    onChange={(e) => setClassFilter(e.target.value)}
                    className="bg-[#1a1a1a] border border-white/10 rounded px-2 py-1 text-[10px] text-neutral-300 font-mono"
                  >
                    <option value="ALL">All Classes</option>
                    {allClasses.map((c) => (
                      <option key={c} value={c}>
                        {c}
                      </option>
                    ))}
                  </select>
                )}
              </div>

              <div className="p-3 space-y-2 max-h-[420px] overflow-y-auto">
                {filteredTracks.map((tr) => {
                  const isSelected = selectedTrackId === tr.track_id;

                  return (
                    <div
                      key={tr.track_id}
                      onClick={() => {
                        setSelectedTrackId(tr.track_id);
                        setCurrentTimeSec(tr.first_timestamp_sec);
                      }}
                      className={`p-3 rounded-lg border cursor-pointer transition-all flex justify-between items-center ${
                        isSelected
                          ? "bg-blue-600/20 border-blue-500/50 text-white"
                          : "bg-[#181818] border-white/5 hover:border-white/20 text-neutral-400"
                      }`}
                    >
                      <div className="space-y-1">
                        <div className="flex items-center gap-2 text-xs font-semibold">
                          <span className="font-mono text-blue-400">Track #{tr.track_id}</span>
                          <span className="capitalize font-mono text-white">{tr.class_name}</span>
                        </div>
                        <div className="text-[10px] text-neutral-500 font-mono">
                          {tr.first_timestamp_sec.toFixed(1)}s → {tr.last_timestamp_sec.toFixed(1)}s (
                          {tr.visibility_duration_sec.toFixed(1)}s)
                        </div>
                      </div>

                      <div className="text-right font-mono text-xs">
                        <div className="text-emerald-400 font-bold">
                          {(tr.avg_confidence * 100).toFixed(0)}% conf
                        </div>
                        <div className="text-[10px] text-neutral-500">
                          {tr.total_distance_px.toFixed(0)} px
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>

            {/* Performance Telemetry Card */}
            {currentRun && (
              <div className="bg-[#121212] border border-white/10 rounded-xl p-5 space-y-3 font-mono text-xs">
                <h3 className="text-xs font-semibold uppercase tracking-wider text-neutral-400 flex items-center gap-2">
                  <Zap className="w-4 h-4 text-amber-400" />
                  Performance & Latency Telemetry
                </h3>
                <div className="space-y-2">
                  <div className="flex justify-between border-b border-white/5 py-1">
                    <span className="text-neutral-400">Processing FPS:</span>
                    <span className="text-emerald-400 font-bold">{currentRun.processing_fps} FPS</span>
                  </div>
                  <div className="flex justify-between border-b border-white/5 py-1">
                    <span className="text-neutral-400">Inference Latency:</span>
                    <span className="text-blue-400">{currentRun.inference_latency_ms} ms/frame</span>
                  </div>
                  <div className="flex justify-between border-b border-white/5 py-1">
                    <span className="text-neutral-400">ByteTrack Latency:</span>
                    <span className="text-purple-400">{currentRun.tracking_latency_ms} ms/frame</span>
                  </div>
                  <div className="flex justify-between py-1">
                    <span className="text-neutral-400">Sampled Frames:</span>
                    <span className="text-white">{currentRun.processed_frames} frames</span>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
