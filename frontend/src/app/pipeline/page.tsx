"use client";

import React, { useEffect, useState } from "react";
import Link from "next/link";
import {
  Activity,
  AlertCircle,
  ArrowRight,
  BarChart2,
  Check,
  CheckCircle2,
  ChevronRight,
  Cpu,
  Database,
  Eye,
  FileCode,
  Flame,
  GitBranch,
  Layers,
  Network,
  Package,
  Play,
  Plus,
  RefreshCw,
  Rocket,
  Search,
  Server,
  Settings,
  ShieldCheck,
  Sparkles,
  Sliders,
  Tag,
  Target,
  Terminal,
  Zap,
} from "lucide-react";
import { PageHeader } from "@/components/ui/PageHeader";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";

// ─── Interfaces ───────────────────────────────────────────────────

type PipelineStage =
  | "DATASET_VERSION"
  | "TRAINING_CONFIG"
  | "TRAINING_RUN"
  | "MODEL_ARTIFACT"
  | "EVALUATION"
  | "BENCHMARK"
  | "MODEL_REGISTRY"
  | "MODEL_COMPARISON"
  | "DEPLOYMENT";

interface StageDetail {
  stage: PipelineStage;
  step_number: number;
  title: string;
  status: "PENDING" | "RUNNING" | "COMPLETED" | "FAILED" | "SKIPPED";
  summary: string;
  metrics: Record<string, any>;
  artifacts: Record<string, any>;
  started_at?: string;
  completed_at?: string;
}

interface ModelLifecyclePipeline {
  pipeline_id: string;
  name: string;
  dataset_id: string;
  dataset_version: string;
  base_model: string;
  target_model_name: string;
  current_stage: PipelineStage;
  status: "PENDING" | "RUNNING" | "COMPLETED" | "FAILED" | "PAUSED";
  stages: Record<string, StageDetail>;
  is_deployed: boolean;
  deployment_endpoint?: string;
  created_at: string;
  completed_at?: string;
}

interface LineageNode {
  id: string;
  stage: string;
  label: string;
  artifact_type: string;
  properties: Record<string, any>;
  parent_node_ids: string[];
}

const STAGES_LIST: { key: PipelineStage; label: string; short: string; icon: any }[] = [
  { key: "DATASET_VERSION", label: "Dataset Version", short: "1. Dataset", icon: Database },
  { key: "TRAINING_CONFIG", label: "Training Config", short: "2. Config", icon: Settings },
  { key: "TRAINING_RUN", label: "Training Run", short: "3. Training", icon: Flame },
  { key: "MODEL_ARTIFACT", label: "Model Artifact", short: "4. Artifact", icon: Package },
  { key: "EVALUATION", label: "Model Evaluation", short: "5. Evaluate", icon: Target },
  { key: "BENCHMARK", label: "Latency Benchmark", short: "6. Benchmark", icon: Zap },
  { key: "MODEL_REGISTRY", label: "Model Registry", short: "7. Registry", icon: Tag },
  { key: "MODEL_COMPARISON", label: "Model Compare", short: "8. Compare", icon: BarChart2 },
  { key: "DEPLOYMENT", label: "Deploy & Inference", short: "9. Deploy", icon: Rocket },
];

export default function ModelLifecyclePipelinePage() {
  const [pipelines, setPipelines] = useState<ModelLifecyclePipeline[]>([]);
  const [selectedPipelineId, setSelectedPipelineId] = useState<string>("");
  const [currentPipeline, setCurrentPipeline] = useState<ModelLifecyclePipeline | null>(null);
  const [selectedStageKey, setSelectedStageKey] = useState<PipelineStage>("DATASET_VERSION");
  const [lineageNodes, setLineageNodes] = useState<LineageNode[]>([]);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [isDeploying, setIsDeploying] = useState<boolean>(false);
  const [toastMessage, setToastMessage] = useState<string | null>(null);

  // New Pipeline Modal
  const [showNewModal, setShowNewModal] = useState<boolean>(false);
  const [newPipeName, setNewPipeName] = useState<string>("Safety PPE Detection Pipeline v2");
  const [newDatasetId, setNewDatasetId] = useState<string>("safety_v2");
  const [newDatasetVersion, setNewDatasetVersion] = useState<string>("v2.1.0");
  const [newBaseModel, setNewBaseModel] = useState<string>("yolo11s.pt");
  const [newTargetName, setNewTargetName] = useState<string>("yolo11s_safety_v2");
  const [newEpochs, setNewEpochs] = useState<number>(50);

  useEffect(() => {
    loadPipelines();
  }, []);

  const loadPipelines = async () => {
    try {
      const res = await fetch("/api/v1/lifecycle/pipelines");
      if (res.ok) {
        const json = await res.json();
        const list: ModelLifecyclePipeline[] = json.data || [];
        setPipelines(list);
        if (list.length > 0 && !selectedPipelineId) {
          setSelectedPipelineId(list[0].pipeline_id);
          setCurrentPipeline(list[0]);
          loadLineage(list[0].pipeline_id);
        }
      }
    } catch (err) {
      console.error("Failed to load pipelines:", err);
    }
  };

  const loadLineage = async (pipeId: string) => {
    try {
      const res = await fetch(`/api/v1/lifecycle/pipelines/${pipeId}/lineage`);
      if (res.ok) {
        const json = await res.json();
        setLineageNodes(json.data || []);
      }
    } catch (err) {
      console.error("Failed to load lineage:", err);
    }
  };

  const handleSelectPipeline = (pipeId: string) => {
    setSelectedPipelineId(pipeId);
    const found = pipelines.find((p) => p.pipeline_id === pipeId);
    if (found) {
      setCurrentPipeline(found);
      loadLineage(pipeId);
    }
  };

  const handleCreatePipeline = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsLoading(true);
    try {
      const res = await fetch("/api/v1/lifecycle/pipelines", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          name: newPipeName,
          dataset_id: newDatasetId,
          dataset_version: newDatasetVersion,
          base_model: newBaseModel,
          target_model_name: newTargetName,
          epochs: newEpochs,
          auto_advance: true,
        }),
      });

      if (res.ok) {
        const json = await res.json();
        setShowNewModal(false);
        showToast(`Created and executed pipeline '${json.data.name}'!`);
        await loadPipelines();
        setSelectedPipelineId(json.data.pipeline_id);
        setCurrentPipeline(json.data);
        loadLineage(json.data.pipeline_id);
      }
    } catch (err) {
      console.error("Failed to create pipeline:", err);
    } finally {
      setIsLoading(false);
    }
  };

  const handleDeployModel = async () => {
    if (!currentPipeline) return;
    setIsDeploying(true);
    try {
      const res = await fetch(`/api/v1/lifecycle/pipelines/${currentPipeline.pipeline_id}/deploy`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          environment: "production",
          device: "auto",
          warm_up_runs: 5,
        }),
      });

      if (res.ok) {
        const json = await res.json();
        setCurrentPipeline(json.data);
        showToast(`Model '${json.data.target_model_name}' deployed to active inference runtime!`);
      }
    } catch (err) {
      console.error("Failed to deploy model:", err);
    } finally {
      setIsDeploying(false);
    }
  };

  const showToast = (msg: string) => {
    setToastMessage(msg);
    setTimeout(() => setToastMessage(null), 3000);
  };

  const activeStageDetail = currentPipeline?.stages?.[selectedStageKey];

  return (
    <div className="space-y-6 pb-16">
      {/* Toast Notification */}
      {toastMessage && (
        <div className="fixed bottom-6 right-6 z-50 flex items-center gap-2 px-4 py-3 bg-emerald-950 border border-emerald-500 text-emerald-200 rounded-lg shadow-xl text-sm animate-fade-in">
          <CheckCircle2 className="w-4 h-4 text-emerald-400" />
          {toastMessage}
        </div>
      )}

      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <PageHeader
            title="Model Production Lifecycle Studio"
            description="End-to-end 9-stage ML production pipeline: Dataset Versioning → Training Config → Run → Artifact → Evaluation → Benchmark → Registry → Compare → Deploy"
          />
        </div>

        <div className="flex items-center gap-3">
          <div className="flex items-center bg-zinc-900 border border-zinc-800 rounded-lg px-3 py-1.5 gap-2">
            <Layers className="w-4 h-4 text-blue-400" />
            <select
              value={selectedPipelineId}
              onChange={(e) => handleSelectPipeline(e.target.value)}
              className="bg-transparent text-sm font-medium text-zinc-200 focus:outline-none"
            >
              {pipelines.map((p) => (
                <option key={p.pipeline_id} value={p.pipeline_id}>
                  {p.name} ({p.status})
                </option>
              ))}
            </select>
          </div>

          <Button
            size="sm"
            onClick={() => setShowNewModal(true)}
            className="gap-1.5 bg-blue-600 hover:bg-blue-500 font-semibold text-xs"
          >
            <Plus className="w-3.5 h-3.5" />
            New Pipeline Run
          </Button>
        </div>
      </div>

      {/* 9-Stage Interactive Stepper & DAG (Step 1-9) */}
      <Card className="bg-zinc-900/60 border-zinc-800 p-4">
        <div className="mb-3 flex items-center justify-between">
          <span className="text-xs font-bold text-zinc-300 uppercase tracking-wider flex items-center gap-2">
            <GitBranch className="w-4 h-4 text-blue-400" />
            Unified Experiment-to-Deployment Pipeline Stepper
          </span>
          <span className="text-xs text-zinc-500 font-mono">
            {currentPipeline?.status === "COMPLETED" ? "All 9 Stages Verified" : "Pipeline Active"}
          </span>
        </div>

        <div className="grid grid-cols-3 sm:grid-cols-5 md:grid-cols-9 gap-2">
          {STAGES_LIST.map((st) => {
            const Icon = st.icon;
            const stageData = currentPipeline?.stages?.[st.key];
            const isCompleted = stageData?.status === "COMPLETED";
            const isSelected = selectedStageKey === st.key;

            return (
              <button
                key={st.key}
                onClick={() => setSelectedStageKey(st.key)}
                className={`p-2.5 rounded-lg border text-left flex flex-col justify-between transition-all ${
                  isSelected
                    ? "bg-blue-600/10 border-blue-500 shadow-md ring-1 ring-blue-500"
                    : isCompleted
                    ? "bg-zinc-900/80 border-emerald-500/30 hover:border-zinc-700"
                    : "bg-zinc-900/40 border-zinc-800 opacity-60 hover:opacity-100"
                }`}
              >
                <div className="flex items-center justify-between mb-2">
                  <Icon
                    className={`w-4 h-4 ${
                      isSelected ? "text-blue-400" : isCompleted ? "text-emerald-400" : "text-zinc-500"
                    }`}
                  />
                  {isCompleted && <Check className="w-3.5 h-3.5 text-emerald-400" />}
                </div>
                <div>
                  <span className="text-[11px] font-bold text-zinc-200 block truncate">{st.short}</span>
                  <span
                    className={`text-[9px] font-semibold uppercase ${
                      isCompleted ? "text-emerald-400" : "text-zinc-500"
                    }`}
                  >
                    {stageData?.status || "PENDING"}
                  </span>
                </div>
              </button>
            );
          })}
        </div>
      </Card>

      {/* Stage Detail Inspector */}
      {activeStageDetail && (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Left 2 Cols: Stage Summary, Metrics, and Artifacts */}
          <div className="lg:col-span-2 space-y-6">
            <Card className="bg-zinc-900/50 border-zinc-800">
              <CardHeader className="pb-3">
                <CardTitle className="text-sm font-semibold text-zinc-200 flex items-center justify-between">
                  <span className="flex items-center gap-2">
                    <Sparkles className="w-4 h-4 text-blue-400" />
                    {activeStageDetail.title}
                  </span>
                  <span className="px-2 py-0.5 text-[10px] font-bold rounded bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
                    {activeStageDetail.status}
                  </span>
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <p className="text-xs text-zinc-300 leading-relaxed bg-zinc-950 p-3 rounded-lg border border-zinc-800">
                  {activeStageDetail.summary}
                </p>

                {/* Metrics Breakdown Grid */}
                <div>
                  <h4 className="text-xs font-bold text-zinc-400 uppercase tracking-wider mb-2">Stage Telemetry Metrics</h4>
                  <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
                    {Object.entries(activeStageDetail.metrics).map(([k, v]) => (
                      <div key={k} className="p-3 bg-zinc-950/80 border border-zinc-800 rounded-lg">
                        <span className="text-[10px] text-zinc-500 uppercase tracking-wider block font-mono">
                          {k.replace(/_/g, " ")}
                        </span>
                        <p className="text-sm font-bold text-zinc-100 mt-0.5">
                          {typeof v === "object" ? JSON.stringify(v) : String(v)}
                        </p>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Artifacts Reference */}
                <div>
                  <h4 className="text-xs font-bold text-zinc-400 uppercase tracking-wider mb-2">Registered Stage Artifacts</h4>
                  <div className="space-y-2">
                    {Object.entries(activeStageDetail.artifacts).map(([k, v]) => (
                      <div
                        key={k}
                        className="flex items-center justify-between p-2.5 bg-zinc-950 rounded-lg border border-zinc-800 text-xs"
                      >
                        <span className="text-zinc-400 font-mono">{k}:</span>
                        <span className="font-mono text-blue-400 truncate max-w-md">
                          {typeof v === "object" ? JSON.stringify(v) : String(v)}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Live Interactive Inference & Deployment Card (Stage 9) */}
            {selectedStageKey === "DEPLOYMENT" && (
              <Card className="bg-zinc-900/50 border-zinc-800">
                <CardHeader>
                  <CardTitle className="text-sm font-semibold text-zinc-200 flex items-center gap-2">
                    <Rocket className="w-4 h-4 text-emerald-400" />
                    Interactive Production Inference Playground
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="aspect-video bg-zinc-950 rounded-lg border border-zinc-800 relative flex items-center justify-center overflow-hidden">
                    <div className="text-center p-4">
                      <Eye className="w-8 h-8 text-zinc-600 mx-auto mb-2" />
                      <p className="text-xs font-mono text-zinc-300">Live Camera Feed: Site CCTV Stream #04</p>
                      <p className="text-[11px] text-zinc-500 mt-1">Inference Engine: {currentPipeline?.target_model_name} (14.8ms / 67.5 FPS)</p>
                    </div>

                    {/* Simulated Detections */}
                    <div
                      className="absolute border-2 border-emerald-500 bg-emerald-500/10 rounded"
                      style={{ top: "30%", left: "35%", width: "25%", height: "45%" }}
                    >
                      <span className="absolute -top-5 left-0 px-1.5 py-0.5 text-[10px] font-bold bg-emerald-600 text-white rounded">
                        person 94%
                      </span>
                    </div>

                    <div
                      className="absolute border-2 border-blue-500 bg-blue-500/10 rounded"
                      style={{ top: "25%", left: "42%", width: "10%", height: "15%" }}
                    >
                      <span className="absolute -top-5 left-0 px-1.5 py-0.5 text-[10px] font-bold bg-blue-600 text-white rounded">
                        helmet 91%
                      </span>
                    </div>
                  </div>

                  <div className="flex items-center justify-between pt-2">
                    <div className="flex items-center gap-2 text-xs text-zinc-400">
                      <span className="flex h-2 w-2 relative">
                        <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
                        <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500"></span>
                      </span>
                      <span>Endpoint Live: <code>{currentPipeline?.deployment_endpoint}</code></span>
                    </div>

                    <div className="flex items-center gap-2">
                      <Link href="/vision-lab">
                        <Button size="sm" variant="outline" className="text-xs h-7 border-blue-500/40 text-blue-200">
                          Open in Vision Lab
                        </Button>
                      </Link>
                      <Link href="/video-lab">
                        <Button size="sm" variant="outline" className="text-xs h-7 border-purple-500/40 text-purple-200">
                          Open in Video Lab
                        </Button>
                      </Link>
                    </div>
                  </div>
                </CardContent>
              </Card>
            )}
          </div>

          {/* Right Col: Provenance Lineage DAG & Quick Actions */}
          <div className="space-y-6">
            <Card className="bg-zinc-900/50 border-zinc-800">
              <CardHeader>
                <CardTitle className="text-sm font-semibold text-zinc-200 flex items-center gap-2">
                  <Network className="w-4 h-4 text-purple-400" />
                  Full Provenance Lineage DAG
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                <p className="text-xs text-zinc-400">
                  Immutable audit trail linking source dataset to production deployment.
                </p>

                <div className="space-y-2 border-l-2 border-blue-500/40 pl-3 ml-2">
                  {lineageNodes.map((node, nIdx) => (
                    <div
                      key={node.id}
                      onClick={() => setSelectedStageKey(node.stage as PipelineStage)}
                      className={`p-2 rounded cursor-pointer transition-colors border text-xs ${
                        selectedStageKey === node.stage
                          ? "bg-blue-600/20 border-blue-500 text-blue-200"
                          : "bg-zinc-950/60 border-zinc-800 hover:border-zinc-700 text-zinc-300"
                      }`}
                    >
                      <div className="flex items-center justify-between">
                        <span className="font-semibold">{node.label}</span>
                      </div>
                      <span className="text-[10px] text-zinc-500 font-mono block mt-0.5">
                        Type: {node.artifact_type}
                      </span>
                    </div>
                  ))}
                </div>

                <div className="pt-3 border-t border-zinc-800">
                  <Button
                    size="sm"
                    onClick={handleDeployModel}
                    disabled={isDeploying || currentPipeline?.is_deployed}
                    className="w-full bg-emerald-600 hover:bg-emerald-500 text-xs font-semibold gap-1.5"
                  >
                    <Rocket className="w-3.5 h-3.5" />
                    {currentPipeline?.is_deployed ? "Model Deployed (Healthy)" : "Deploy Model to Production"}
                  </Button>
                </div>
              </CardContent>
            </Card>
          </div>
        </div>
      )}

      {/* New Pipeline Modal */}
      {showNewModal && (
        <div className="fixed inset-0 z-50 bg-black/80 backdrop-blur-md flex items-center justify-center p-4">
          <div className="bg-zinc-900 border border-zinc-800 rounded-xl w-full max-w-lg shadow-2xl overflow-hidden animate-scale-in">
            <div className="p-4 border-b border-zinc-800 bg-zinc-950/80 flex items-center justify-between">
              <span className="text-sm font-bold text-zinc-200 flex items-center gap-2">
                <Plus className="w-4 h-4 text-blue-400" />
                Create New Model Lifecycle Pipeline
              </span>
              <button onClick={() => setShowNewModal(false)} className="text-zinc-500 hover:text-zinc-300">
                &times;
              </button>
            </div>

            <form onSubmit={handleCreatePipeline} className="p-6 space-y-4">
              <div>
                <label className="text-xs text-zinc-400 block mb-1">Pipeline Name</label>
                <input
                  type="text"
                  value={newPipeName}
                  onChange={(e) => setNewPipeName(e.target.value)}
                  className="w-full bg-zinc-950 border border-zinc-800 rounded px-3 py-1.5 text-xs text-zinc-200"
                  required
                />
              </div>

              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="text-xs text-zinc-400 block mb-1">Dataset ID</label>
                  <input
                    type="text"
                    value={newDatasetId}
                    onChange={(e) => setNewDatasetId(e.target.value)}
                    className="w-full bg-zinc-950 border border-zinc-800 rounded px-3 py-1.5 text-xs text-zinc-200"
                    required
                  />
                </div>
                <div>
                  <label className="text-xs text-zinc-400 block mb-1">Dataset Version</label>
                  <input
                    type="text"
                    value={newDatasetVersion}
                    onChange={(e) => setNewDatasetVersion(e.target.value)}
                    className="w-full bg-zinc-950 border border-zinc-800 rounded px-3 py-1.5 text-xs text-zinc-200"
                    required
                  />
                </div>
              </div>

              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="text-xs text-zinc-400 block mb-1">Base Pretrained Model</label>
                  <input
                    type="text"
                    value={newBaseModel}
                    onChange={(e) => setNewBaseModel(e.target.value)}
                    className="w-full bg-zinc-950 border border-zinc-800 rounded px-3 py-1.5 text-xs text-zinc-200"
                    required
                  />
                </div>
                <div>
                  <label className="text-xs text-zinc-400 block mb-1">Target Registered Name</label>
                  <input
                    type="text"
                    value={newTargetName}
                    onChange={(e) => setNewTargetName(e.target.value)}
                    className="w-full bg-zinc-950 border border-zinc-800 rounded px-3 py-1.5 text-xs text-zinc-200"
                    required
                  />
                </div>
              </div>

              <div>
                <label className="text-xs text-zinc-400 block mb-1">Training Epochs</label>
                <input
                  type="number"
                  value={newEpochs}
                  onChange={(e) => setNewEpochs(Number(e.target.value))}
                  min={1}
                  max={500}
                  className="w-full bg-zinc-950 border border-zinc-800 rounded px-3 py-1.5 text-xs text-zinc-200"
                  required
                />
              </div>

              <div className="pt-4 border-t border-zinc-800 flex items-center justify-end gap-3">
                <Button size="sm" variant="ghost" type="button" onClick={() => setShowNewModal(false)} className="text-xs">
                  Cancel
                </Button>
                <Button
                  size="sm"
                  type="submit"
                  disabled={isLoading}
                  className="bg-blue-600 hover:bg-blue-500 text-xs font-semibold gap-1.5"
                >
                  <Play className="w-3.5 h-3.5" />
                  Execute Full Pipeline
                </Button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
}
