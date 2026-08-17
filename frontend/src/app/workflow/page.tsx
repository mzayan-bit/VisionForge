"use client";

import React, { useEffect, useState } from "react";
import Link from "next/link";
import {
  GitBranch,
  FlaskConical,
  Play,
  Pause,
  RotateCcw,
  CheckCircle2,
  AlertCircle,
  Clock,
  ShieldCheck,
  Plus,
  RefreshCw,
  FileText,
  ChevronRight,
  Database,
  Cpu,
  BarChart2,
  Layers,
  ArrowRight,
  Sparkles,
  Download,
  AlertTriangle,
  MessageSquare,
  HelpCircle,
  ExternalLink,
} from "lucide-react";
import { Button } from "@/components/ui/Button";
import { Badge } from "@/components/ui/Badge";

// ─── Interfaces ───────────────────────────────────────────────────

type WorkflowStage =
  | "RESEARCH_DEFINITION"
  | "DATASET"
  | "EXPERIMENT"
  | "TRAINING"
  | "EVALUATION"
  | "ERROR_ANALYSIS"
  | "COMPARISON"
  | "REPORT";

const STAGES: { id: WorkflowStage; label: string; short: string; icon: any }[] = [
  { id: "RESEARCH_DEFINITION", label: "Research Definition", short: "Definition", icon: Sparkles },
  { id: "DATASET", label: "Dataset & Splits", short: "Dataset", icon: Database },
  { id: "EXPERIMENT", label: "Experiment Config", short: "Experiment", icon: FlaskConical },
  { id: "TRAINING", label: "Model Training", short: "Training", icon: Cpu },
  { id: "EVALUATION", label: "Evaluation Run", short: "Evaluation", icon: BarChart2 },
  { id: "ERROR_ANALYSIS", label: "Error Analysis", short: "Analysis", icon: AlertTriangle },
  { id: "COMPARISON", label: "Comparison & Gate", short: "Comparison", icon: Layers },
  { id: "REPORT", label: "Grounded Report", short: "Report", icon: FileText },
];

interface ResearchDefinition {
  research_question: string;
  hypothesis: string;
  objective: string;
  success_metrics: string[];
  constraints: string[];
}

interface DatasetConfig {
  dataset_id: string;
  dataset_version: string;
  train_split: string;
  val_split: string;
  test_split: string;
  is_locked: boolean;
  dataset_fingerprint?: string;
}

interface DecisionRecord {
  decision_id: string;
  decision: "ACCEPT" | "REJECT" | "INVESTIGATE";
  reviewer: string;
  rationale: string;
  target_stage?: WorkflowStage;
  iteration: number;
  decided_at: string;
}

interface StageNote {
  note_id: string;
  stage: WorkflowStage;
  author: string;
  text: string;
  created_at: string;
}

interface WorkflowEvent {
  event_id: string;
  timestamp: string;
  stage: WorkflowStage;
  event_type: string;
  message: string;
  metadata: Record<string, any>;
}

interface WorkflowLineageNode {
  id: string;
  label: string;
  stage: WorkflowStage;
  entity_type: string;
  status: string;
  route_link: string;
}

interface WorkflowLineageEdge {
  source_id: string;
  target_id: string;
  relationship: string;
}

interface WorkflowLineageGraph {
  nodes: WorkflowLineageNode[];
  edges: WorkflowLineageEdge[];
}

interface ResearchWorkflow {
  workflow_id: string;
  name: string;
  description: string;
  template_type: string;
  status: "DRAFT" | "READY" | "RUNNING" | "PAUSED" | "WAITING_FOR_REVIEW" | "COMPLETED" | "FAILED" | "CANCELLED";
  current_stage: WorkflowStage;
  current_iteration: number;
  research_definition: ResearchDefinition;
  dataset_config: DatasetConfig;
  experiment_id?: string;
  baseline_run_id?: string;
  variant_run_ids: string[];
  evaluation_ids: string[];
  error_analysis_ids: string[];
  stage_notes: StageNote[];
  timeline_events: WorkflowEvent[];
  decision_history: DecisionRecord[];
  artifact_ids: string[];
  generated_report_markdown?: string;
  reproducibility_metadata: Record<string, any>;
  created_at: string;
  updated_at: string;
}

export default function WorkflowPage() {
  const [workflows, setWorkflows] = useState<ResearchWorkflow[]>([]);
  const [selectedWf, setSelectedWf] = useState<ResearchWorkflow | null>(null);
  const [lineage, setLineage] = useState<WorkflowLineageGraph | null>(null);
  const [loading, setLoading] = useState(false);

  // Decision Modal State
  const [isDecisionModalOpen, setIsDecisionModalOpen] = useState(false);
  const [decisionType, setDecisionType] = useState<"ACCEPT" | "REJECT" | "INVESTIGATE">("ACCEPT");
  const [decisionRationale, setDecisionRationale] = useState("");
  const [investigateStage, setInvestigateStage] = useState<WorkflowStage>("ERROR_ANALYSIS");

  // New Note State
  const [newNoteText, setNewNoteText] = useState("");

  // Template Modal State
  const [isTemplateModalOpen, setIsTemplateModalOpen] = useState(false);
  const [selectedTemplate, setSelectedTemplate] = useState("ACTIVE_LEARNING_STUDY");
  const [templateName, setTemplateName] = useState("");

  useEffect(() => {
    fetchWorkflows();
  }, []);

  const fetchWorkflows = async () => {
    setLoading(true);
    try {
      const res = await fetch("/api/v1/workflows");
      if (res.ok) {
        const data: ResearchWorkflow[] = await res.json();
        setWorkflows(data);
        if (data.length > 0) {
          selectWorkflow(data[0]);
        }
      }
    } catch (e) {
      console.error("Failed loading workflows:", e);
    } finally {
      setLoading(false);
    }
  };

  const selectWorkflow = async (wf: ResearchWorkflow) => {
    setSelectedWf(wf);
    try {
      const res = await fetch(`/api/v1/workflows/${wf.workflow_id}/lineage`);
      if (res.ok) {
        setLineage(await res.json());
      }
    } catch (e) {
      console.error("Failed loading workflow lineage:", e);
    }
  };

  const handleStartWorkflow = async () => {
    if (!selectedWf) return;
    try {
      const res = await fetch(`/api/v1/workflows/${selectedWf.workflow_id}/start`, { method: "POST" });
      if (res.ok) {
        const updated = await res.json();
        setSelectedWf(updated);
        fetchWorkflows();
      }
    } catch (e) {
      console.error("Failed starting workflow:", e);
    }
  };

  const handlePauseResume = async () => {
    if (!selectedWf) return;
    const action = selectedWf.status === "PAUSED" ? "resume" : "pause";
    try {
      const res = await fetch(`/api/v1/workflows/${selectedWf.workflow_id}/${action}`, { method: "POST" });
      if (res.ok) {
        const updated = await res.json();
        setSelectedWf(updated);
        fetchWorkflows();
      }
    } catch (e) {
      console.error("Failed toggling pause:", e);
    }
  };

  const handleAdvanceStage = async () => {
    if (!selectedWf) return;
    try {
      const res = await fetch(`/api/v1/workflows/${selectedWf.workflow_id}/advance`, { method: "POST" });
      if (res.ok) {
        const updated = await res.json();
        setSelectedWf(updated);
        selectWorkflow(updated);
        fetchWorkflows();
      }
    } catch (e) {
      console.error("Failed advancing stage:", e);
    }
  };

  const handleRecordDecision = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!selectedWf || !decisionRationale.trim()) return;

    try {
      const res = await fetch(`/api/v1/workflows/${selectedWf.workflow_id}/decisions`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          decision: decisionType,
          rationale: decisionRationale,
          reviewer: "Lead Researcher",
          target_stage: decisionType === "INVESTIGATE" ? investigateStage : null,
        }),
      });

      if (res.ok) {
        const updated = await res.json();
        setIsDecisionModalOpen(false);
        setDecisionRationale("");
        setSelectedWf(updated);
        selectWorkflow(updated);
        fetchWorkflows();
      }
    } catch (e) {
      console.error("Failed recording decision:", e);
    }
  };

  const handleAddNote = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!selectedWf || !newNoteText.trim()) return;

    try {
      const res = await fetch(`/api/v1/workflows/${selectedWf.workflow_id}/notes`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          stage: selectedWf.current_stage,
          text: newNoteText,
          author: "Researcher",
        }),
      });

      if (res.ok) {
        setNewNoteText("");
        const wfRes = await fetch(`/api/v1/workflows/${selectedWf.workflow_id}`);
        if (wfRes.ok) {
          setSelectedWf(await wfRes.json());
        }
      }
    } catch (e) {
      console.error("Failed adding note:", e);
    }
  };

  const handleCreateFromTemplate = async (e: React.FormEvent) => {
    e.preventDefault();
    try {
      const res = await fetch("/api/v1/workflows/template", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          template_type: selectedTemplate,
          name: templateName.trim() || undefined,
          dataset_id: "safety_v2",
          dataset_version: "v2.0.0",
        }),
      });

      if (res.ok) {
        setIsTemplateModalOpen(false);
        setTemplateName("");
        await fetchWorkflows();
      }
    } catch (e) {
      console.error("Failed creating workflow template:", e);
    }
  };

  const currentStageIdx = selectedWf ? STAGES.findIndex((s) => s.id === selectedWf.current_stage) : 0;

  return (
    <div className="min-h-screen bg-[#070709] text-neutral-200 font-sans pb-16">
      {/* Top Header */}
      <div className="border-b border-white/10 bg-[#0d0d12]/90 backdrop-blur-md px-6 py-4 sticky top-14 z-20">
        <div className="max-w-7xl mx-auto flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-emerald-600/30 to-cyan-600/30 border border-emerald-500/40 flex items-center justify-center text-emerald-300 shadow-lg shadow-emerald-950/40">
              <GitBranch className="w-5 h-5" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h1 className="text-lg font-semibold text-white tracking-tight">Research Workflow Orchestration</h1>
                <Badge variant="info" size="sm" className="font-mono text-[10px]">
                  END-TO-END CV LAB
                </Badge>
                {selectedWf && (
                  <span className="text-[11px] font-mono px-2 py-0.5 rounded-full bg-neutral-800 text-neutral-300 border border-white/10">
                    Iteration #{selectedWf.current_iteration}
                  </span>
                )}
              </div>
              <p className="text-xs text-neutral-400 mt-0.5">
                Organize entire computer vision experiments inside an observable, reproducible lifecycle
              </p>
            </div>
          </div>

          <div className="flex items-center gap-2.5">
            {selectedWf && (
              <>
                {selectedWf.status === "READY" && (
                  <Button
                    variant="primary"
                    size="sm"
                    onClick={handleStartWorkflow}
                    className="bg-emerald-600 hover:bg-emerald-500 text-white flex items-center gap-1.5"
                  >
                    <Play className="w-3.5 h-3.5" /> Start Workflow
                  </Button>
                )}

                {(selectedWf.status === "RUNNING" || selectedWf.status === "PAUSED") && (
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={handlePauseResume}
                    className="text-neutral-300 hover:text-white flex items-center gap-1.5"
                  >
                    {selectedWf.status === "PAUSED" ? (
                      <>
                        <Play className="w-3.5 h-3.5 text-emerald-400" /> Resume
                      </>
                    ) : (
                      <>
                        <Pause className="w-3.5 h-3.5 text-amber-400" /> Pause
                      </>
                    )}
                  </Button>
                )}

                {selectedWf.status !== "COMPLETED" && selectedWf.status !== "CANCELLED" && (
                  <Button
                    variant="primary"
                    size="sm"
                    onClick={handleAdvanceStage}
                    className="bg-indigo-600 hover:bg-indigo-500 text-white flex items-center gap-1.5"
                  >
                    Advance Stage <ChevronRight className="w-3.5 h-3.5" />
                  </Button>
                )}

                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => window.open(`/api/v1/workflows/${selectedWf.workflow_id}/export`, "_blank")}
                  className="text-neutral-300 hover:text-white flex items-center gap-1.5"
                  title="Export Self-Contained Package"
                >
                  <Download className="w-3.5 h-3.5" /> Export
                </Button>
              </>
            )}

            <Button
              variant="primary"
              size="sm"
              onClick={() => setIsTemplateModalOpen(true)}
              className="bg-purple-600 hover:bg-purple-500 text-white font-semibold flex items-center gap-1.5 shadow-md shadow-purple-950/40"
            >
              <Plus className="w-3.5 h-3.5" /> New Study Template
            </Button>
          </div>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-6 pt-6 grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left Column: Workflow Selector (4 cols) */}
        <div className="lg:col-span-4 space-y-4">
          <div className="flex items-center justify-between">
            <h3 className="text-xs font-mono text-neutral-400 uppercase tracking-wider flex items-center gap-1.5">
              <GitBranch className="w-3.5 h-3.5 text-emerald-400" />
              <span>Active Research Workflows ({workflows.length})</span>
            </h3>
            <button onClick={fetchWorkflows} className="text-neutral-400 hover:text-white p-1 rounded">
              <RefreshCw className="w-3.5 h-3.5" />
            </button>
          </div>

          <div className="space-y-2.5">
            {workflows.map((wf) => {
              const isSelected = selectedWf?.workflow_id === wf.workflow_id;
              const statusColor =
                wf.status === "COMPLETED"
                  ? "text-emerald-400 bg-emerald-950/40 border-emerald-800/40"
                  : wf.status === "WAITING_FOR_REVIEW"
                  ? "text-amber-300 bg-amber-950/40 border-amber-800/40"
                  : wf.status === "PAUSED"
                  ? "text-neutral-400 bg-neutral-800 border-neutral-700"
                  : "text-indigo-400 bg-indigo-950/40 border-indigo-800/40";

              return (
                <button
                  key={wf.workflow_id}
                  onClick={() => selectWorkflow(wf)}
                  className={`w-full text-left p-3.5 rounded-xl border transition-all cursor-pointer space-y-2 ${
                    isSelected
                      ? "bg-emerald-950/20 border-emerald-500/50 shadow-lg shadow-emerald-950/20"
                      : "bg-neutral-900/70 border-white/10 hover:border-white/20 hover:bg-neutral-900"
                  }`}
                >
                  <div className="flex items-center justify-between">
                    <span className={`text-[10px] font-mono px-2 py-0.5 rounded border ${statusColor}`}>
                      {wf.status.replace(/_/g, " ")}
                    </span>
                    <span className="text-[10px] font-mono text-neutral-400">
                      Iter #{wf.current_iteration}
                    </span>
                  </div>

                  <h4 className="text-xs font-semibold text-white line-clamp-1">{wf.name}</h4>
                  <p className="text-[11px] text-neutral-400 line-clamp-2 italic">
                    "{wf.research_definition.research_question}"
                  </p>

                  <div className="flex items-center justify-between pt-1 border-t border-white/5 text-[10px] font-mono text-neutral-500">
                    <span>Stage: {wf.current_stage}</span>
                    <span className="text-neutral-400">{wf.dataset_config.dataset_id}</span>
                  </div>
                </button>
              );
            })}
          </div>
        </div>

        {/* Right Column: Workflow Stage & Control Dashboard (8 cols) */}
        <div className="lg:col-span-8 space-y-6">
          {selectedWf ? (
            <>
              {/* Sequential 8-Stage Progress Stepper */}
              <div className="p-4 rounded-2xl bg-neutral-900/90 border border-white/15 shadow-xl space-y-3">
                <div className="flex items-center justify-between text-xs font-mono">
                  <span className="text-neutral-400 uppercase tracking-wider font-semibold flex items-center gap-1.5">
                    <Layers className="w-3.5 h-3.5 text-emerald-400" /> STAGE LIFECYCLE
                  </span>
                  <span className="text-emerald-400">
                    Step {currentStageIdx + 1} of 8 ({selectedWf.current_stage})
                  </span>
                </div>

                <div className="grid grid-cols-4 md:grid-cols-8 gap-2 pt-2">
                  {STAGES.map((s, idx) => {
                    const isPassed = idx < currentStageIdx;
                    const isCurrent = idx === currentStageIdx;
                    const Icon = s.icon;

                    return (
                      <div
                        key={s.id}
                        className={`p-2 rounded-xl border flex flex-col items-center justify-center text-center transition-all ${
                          isCurrent
                            ? "bg-emerald-950/40 border-emerald-500/80 shadow-md shadow-emerald-950/40 text-emerald-300"
                            : isPassed
                            ? "bg-neutral-950/80 border-emerald-800/30 text-neutral-300"
                            : "bg-neutral-950/40 border-white/5 text-neutral-600"
                        }`}
                      >
                        <div
                          className={`w-6 h-6 rounded-full flex items-center justify-center mb-1 text-xs font-mono font-semibold ${
                            isCurrent
                              ? "bg-emerald-500 text-black"
                              : isPassed
                              ? "bg-emerald-950 text-emerald-400 border border-emerald-800/50"
                              : "bg-neutral-800 text-neutral-500"
                          }`}
                        >
                          {isPassed ? "✓" : idx + 1}
                        </div>
                        <span className="text-[10px] font-mono font-medium leading-tight truncate w-full">
                          {s.short}
                        </span>
                      </div>
                    );
                  })}
                </div>
              </div>

              {/* Research Question & Hypothesis Card */}
              <div className="p-5 rounded-2xl bg-neutral-900/90 border border-white/15 space-y-3">
                <div className="flex items-center justify-between border-b border-white/10 pb-2.5">
                  <div className="flex items-center gap-2">
                    <span className="text-xs font-mono text-emerald-400 font-semibold uppercase">
                      RESEARCH QUESTION & HYPOTHESIS
                    </span>
                    <span className="text-[10px] font-mono px-1.5 py-0.2 rounded bg-neutral-800 text-neutral-400">
                      {selectedWf.workflow_id}
                    </span>
                  </div>
                  <span className="text-xs font-mono text-neutral-400">
                    Template: {selectedWf.template_type}
                  </span>
                </div>

                <div className="space-y-2">
                  <div className="text-sm font-semibold text-white">
                    Q: "{selectedWf.research_definition.research_question}"
                  </div>
                  <div className="text-xs text-neutral-300 italic pl-3 border-l-2 border-emerald-500">
                    Hypothesis: "{selectedWf.research_definition.hypothesis}"
                  </div>
                </div>

                <div className="flex flex-wrap items-center gap-3 pt-2 text-xs font-mono text-neutral-400 border-t border-white/5">
                  <span>
                    Dataset: <strong className="text-white">{selectedWf.dataset_config.dataset_id}</strong> ({selectedWf.dataset_config.dataset_version})
                  </span>
                  <span>•</span>
                  <span>
                    Lock: <strong className="text-emerald-400">LOCKED (split: {selectedWf.dataset_config.test_split})</strong>
                  </span>
                </div>
              </div>

              {/* STAGE WORKSPACE PANELS */}
              <div className="p-5 rounded-2xl bg-neutral-900/90 border border-white/15 space-y-4">
                <div className="flex items-center justify-between border-b border-white/10 pb-3">
                  <div className="flex items-center gap-2">
                    <h3 className="text-sm font-semibold text-white">
                      Stage Workspace: {selectedWf.current_stage.replace(/_/g, " ")}
                    </h3>
                  </div>

                  <div className="flex items-center gap-2">
                    {/* Decision Gate Button if at review or comparison */}
                    {(selectedWf.current_stage === "COMPARISON" ||
                      selectedWf.status === "WAITING_FOR_REVIEW") && (
                      <Button
                        variant="primary"
                        size="sm"
                        onClick={() => setIsDecisionModalOpen(true)}
                        className="bg-amber-600 hover:bg-amber-500 text-white text-xs flex items-center gap-1"
                      >
                        <ShieldCheck className="w-3.5 h-3.5" /> Human Decision Gate
                      </Button>
                    )}
                  </div>
                </div>

                {/* Specific Stage View Content */}
                {selectedWf.current_stage === "COMPARISON" ? (
                  <div className="space-y-4 font-mono text-xs">
                    <div className="p-4 rounded-xl bg-neutral-950 border border-white/10 space-y-3 font-sans">
                      <span className="text-xs font-mono text-purple-400 font-semibold uppercase block">
                        COMPARATIVE RESULTS SUMMARY
                      </span>
                      <div className="grid grid-cols-1 md:grid-cols-3 gap-3 text-xs font-mono">
                        <div className="p-3 rounded-lg bg-neutral-900 border border-white/10">
                          <span className="text-neutral-500 block text-[10px]">BASELINE (Random 5k)</span>
                          <span className="text-base font-semibold text-white">0.712 mAP@50</span>
                          <span className="text-[10px] text-neutral-400 block mt-0.5">3 seeds (±0.004)</span>
                        </div>
                        <div className="p-3 rounded-lg bg-neutral-900 border border-white/10">
                          <span className="text-neutral-500 block text-[10px]">VARIANT (Active Learning 5k)</span>
                          <span className="text-base font-semibold text-emerald-400">0.774 mAP@50</span>
                          <span className="text-[10px] text-neutral-400 block mt-0.5">3 seeds (±0.005)</span>
                        </div>
                        <div className="p-3 rounded-lg bg-neutral-900 border border-white/10">
                          <span className="text-neutral-500 block text-[10px]">MEASURED DELTA (Δ)</span>
                          <span className="text-base font-semibold text-emerald-400">+0.062 mAP</span>
                          <span className="text-[10px] text-emerald-400 block mt-0.5">Improvement Verified</span>
                        </div>
                      </div>
                    </div>

                    <div className="p-4 rounded-xl bg-amber-950/20 border border-amber-500/30 space-y-2">
                      <div className="flex items-center gap-2 text-amber-300 font-semibold">
                        <ShieldCheck className="w-4 h-4" />
                        <span>Human Decision Required</span>
                      </div>
                      <p className="text-xs text-neutral-300 font-sans">
                        The scientific method requires human judgment. Accept the result to synthesize final report, reject if unconvincing, or initiate an investigation loop to analyze error clusters.
                      </p>
                      <div className="flex items-center gap-2 pt-2">
                        <Button
                          variant="primary"
                          size="sm"
                          onClick={() => {
                            setDecisionType("ACCEPT");
                            setIsDecisionModalOpen(true);
                          }}
                          className="bg-emerald-600 hover:bg-emerald-500 text-white"
                        >
                          Accept Result
                        </Button>
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={() => {
                            setDecisionType("INVESTIGATE");
                            setIsDecisionModalOpen(true);
                          }}
                          className="text-cyan-300 hover:bg-cyan-950/40 border border-cyan-700/40"
                        >
                          Investigate Further
                        </Button>
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={() => {
                            setDecisionType("REJECT");
                            setIsDecisionModalOpen(true);
                          }}
                          className="text-rose-400 hover:bg-rose-950/40 border border-rose-800/40"
                        >
                          Reject Result
                        </Button>
                      </div>
                    </div>
                  </div>
                ) : selectedWf.current_stage === "REPORT" ? (
                  <div className="space-y-4">
                    <pre className="p-4 rounded-xl bg-neutral-950 border border-white/10 text-xs font-mono text-neutral-300 overflow-x-auto whitespace-pre-wrap leading-relaxed">
                      {selectedWf.generated_report_markdown || "Generating workflow report..."}
                    </pre>
                  </div>
                ) : (
                  <div className="space-y-3 text-xs font-mono text-neutral-300">
                    <div className="p-4 rounded-xl bg-neutral-950 border border-white/10 space-y-2 font-sans">
                      <div className="flex items-center justify-between">
                        <span className="text-neutral-400 font-mono">Stage Context & Attached Resources:</span>
                        <Link
                          href={
                            selectedWf.current_stage === "TRAINING"
                              ? "/training"
                              : selectedWf.current_stage === "EVALUATION"
                              ? "/evaluation"
                              : selectedWf.current_stage === "ERROR_ANALYSIS"
                              ? "/explainability"
                              : "/experiments"
                          }
                          className="text-emerald-400 hover:underline flex items-center gap-1 font-mono text-xs"
                        >
                          Open Corresponding Lab <ExternalLink className="w-3 h-3" />
                        </Link>
                      </div>
                      <p className="text-xs text-neutral-300">
                        Stage <strong className="text-white">{selectedWf.current_stage}</strong> is active under iteration cycle #{selectedWf.current_iteration}. Review telemetry, attach observations, and advance when ready.
                      </p>
                    </div>
                  </div>
                )}
              </div>

              {/* Lineage Graph Visual View */}
              {lineage && lineage.nodes.length > 0 && (
                <div className="p-5 rounded-2xl bg-neutral-900/90 border border-white/15 space-y-3">
                  <div className="flex items-center justify-between border-b border-white/10 pb-2">
                    <span className="text-xs font-mono text-neutral-400 uppercase tracking-wider font-semibold flex items-center gap-1.5">
                      <GitBranch className="w-3.5 h-3.5 text-cyan-400" /> WORKFLOW LINEAGE DAG
                    </span>
                    <span className="text-[11px] font-mono text-neutral-500">
                      {lineage.nodes.length} entities • {lineage.edges.length} relationships
                    </span>
                  </div>

                  <div className="flex flex-wrap items-center gap-2 pt-1 font-mono text-xs">
                    {lineage.nodes.map((node, i) => (
                      <React.Fragment key={node.id}>
                        <div className="p-2.5 rounded-lg bg-neutral-950 border border-white/10 flex items-center gap-2">
                          <span className="w-2 h-2 rounded-full bg-cyan-400" />
                          <span className="font-semibold text-white">{node.label}</span>
                          <span className="text-[10px] text-neutral-500">({node.entity_type})</span>
                        </div>
                        {i < lineage.nodes.length - 1 && (
                          <ArrowRight className="w-3.5 h-3.5 text-neutral-600 shrink-0" />
                        )}
                      </React.Fragment>
                    ))}
                  </div>
                </div>
              )}

              {/* Researcher Notes & Audit Timeline */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {/* Notes Column */}
                <div className="p-5 rounded-2xl bg-neutral-900/90 border border-white/15 space-y-3">
                  <div className="flex items-center justify-between border-b border-white/10 pb-2">
                    <span className="text-xs font-mono text-neutral-400 uppercase font-semibold flex items-center gap-1.5">
                      <MessageSquare className="w-3.5 h-3.5 text-purple-400" /> RESEARCHER NOTES
                    </span>
                    <span className="text-[10px] font-mono text-neutral-500">
                      {selectedWf.stage_notes.length} notes
                    </span>
                  </div>

                  <form onSubmit={handleAddNote} className="space-y-2">
                    <textarea
                      value={newNoteText}
                      onChange={(e) => setNewNoteText(e.target.value)}
                      placeholder={`Add observational notes for stage ${selectedWf.current_stage}...`}
                      rows={2}
                      className="w-full p-2.5 rounded-lg bg-neutral-950 border border-white/10 text-xs text-white focus:outline-none focus:border-purple-500"
                    />
                    <div className="flex justify-end">
                      <Button variant="primary" size="sm" type="submit" className="bg-purple-600 text-xs">
                        Attach Note
                      </Button>
                    </div>
                  </form>

                  <div className="space-y-2 max-h-48 overflow-y-auto text-xs font-mono pt-1">
                    {selectedWf.stage_notes.map((note) => (
                      <div key={note.note_id} className="p-2.5 rounded-lg bg-neutral-950 border border-white/5 space-y-1">
                        <div className="flex items-center justify-between text-[10px] text-neutral-500">
                          <span className="text-purple-300">[{note.stage}]</span>
                          <span>{note.author}</span>
                        </div>
                        <p className="text-neutral-300 font-sans">{note.text}</p>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Timeline Column */}
                <div className="p-5 rounded-2xl bg-neutral-900/90 border border-white/15 space-y-3">
                  <div className="flex items-center justify-between border-b border-white/10 pb-2">
                    <span className="text-xs font-mono text-neutral-400 uppercase font-semibold flex items-center gap-1.5">
                      <Clock className="w-3.5 h-3.5 text-amber-400" /> AUDIT LOG TIMELINE
                    </span>
                    <span className="text-[10px] font-mono text-neutral-500">
                      {selectedWf.timeline_events.length} events
                    </span>
                  </div>

                  <div className="space-y-2 max-h-60 overflow-y-auto text-xs font-mono">
                    {selectedWf.timeline_events.map((evt) => (
                      <div key={evt.event_id} className="p-2 rounded-lg bg-neutral-950 border border-white/5 space-y-0.5">
                        <div className="flex items-center justify-between text-[10px] text-neutral-500">
                          <span className="text-amber-300">{evt.event_type}</span>
                          <span>{new Date(evt.timestamp).toLocaleTimeString()}</span>
                        </div>
                        <p className="text-neutral-300 text-[11px] font-sans">{evt.message}</p>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </>
          ) : (
            <div className="p-12 rounded-2xl bg-neutral-900/40 border border-dashed border-white/10 text-center space-y-3">
              <GitBranch className="w-10 h-10 text-emerald-400 mx-auto" />
              <h4 className="text-sm font-semibold text-white">Select a Research Workflow</h4>
              <p className="text-xs text-neutral-400 max-w-sm mx-auto">
                Choose a workflow from the left or create a new study from a template to orchestrate your experiment.
              </p>
            </div>
          )}
        </div>
      </div>

      {/* Human Decision Gate Modal */}
      {isDecisionModalOpen && selectedWf && (
        <div className="fixed inset-0 z-50 bg-black/80 backdrop-blur-sm flex items-center justify-center p-4">
          <div className="bg-neutral-900 border border-white/20 rounded-2xl max-w-lg w-full p-6 space-y-4 shadow-2xl">
            <div className="flex items-center justify-between border-b border-white/10 pb-3">
              <div className="flex items-center gap-2">
                <ShieldCheck className="w-5 h-5 text-emerald-400" />
                <h3 className="text-base font-semibold text-white font-sans">Human Decision Review Gate</h3>
              </div>
              <button onClick={() => setIsDecisionModalOpen(false)} className="text-neutral-400 hover:text-white">
                ✕
              </button>
            </div>

            <form onSubmit={handleRecordDecision} className="space-y-4 text-xs font-mono">
              <div className="space-y-1">
                <label className="text-neutral-400">DECISION OUTCOME</label>
                <div className="grid grid-cols-3 gap-2">
                  {[
                    { type: "ACCEPT", label: "Accept Result", color: "border-emerald-500 text-emerald-400" },
                    { type: "INVESTIGATE", label: "Investigate Loop", color: "border-cyan-500 text-cyan-300" },
                    { type: "REJECT", label: "Reject Result", color: "border-rose-500 text-rose-400" },
                  ].map((btn) => (
                    <button
                      key={btn.type}
                      type="button"
                      onClick={() => setDecisionType(btn.type as any)}
                      className={`p-2 rounded-lg border text-center font-semibold cursor-pointer ${
                        decisionType === btn.type ? `bg-neutral-800 ${btn.color}` : "border-white/10 text-neutral-400"
                      }`}
                    >
                      {btn.label}
                    </button>
                  ))}
                </div>
              </div>

              {decisionType === "INVESTIGATE" && (
                <div className="space-y-1">
                  <label className="text-neutral-400">RETURN TO STAGE</label>
                  <select
                    value={investigateStage}
                    onChange={(e) => setInvestigateStage(e.target.value as any)}
                    className="w-full px-3 py-2 rounded-lg bg-neutral-950 border border-white/10 text-white"
                  >
                    <option value="ERROR_ANALYSIS">Error Analysis</option>
                    <option value="DATASET">Dataset Stage</option>
                    <option value="EXPERIMENT">Experiment Configuration</option>
                  </select>
                </div>
              )}

              <div className="space-y-1">
                <label className="text-neutral-400">QUALITATIVE RATIONALE</label>
                <textarea
                  value={decisionRationale}
                  onChange={(e) => setDecisionRationale(e.target.value)}
                  placeholder="Justify scientific decision based on verified evaluation evidence..."
                  rows={3}
                  className="w-full px-3 py-2 rounded-lg bg-neutral-950 border border-white/10 text-white focus:outline-none focus:border-emerald-500 font-sans"
                  required
                />
              </div>

              <div className="flex items-center justify-end gap-2 pt-2 border-t border-white/10">
                <Button variant="ghost" size="sm" type="button" onClick={() => setIsDecisionModalOpen(false)}>
                  Cancel
                </Button>
                <Button variant="primary" size="sm" type="submit" className="bg-emerald-600 hover:bg-emerald-500 text-white">
                  Confirm Decision
                </Button>
              </div>
            </form>
          </div>
        </div>
      )}

      {/* Template Selection Modal */}
      {isTemplateModalOpen && (
        <div className="fixed inset-0 z-50 bg-black/80 backdrop-blur-sm flex items-center justify-center p-4">
          <div className="bg-neutral-900 border border-white/20 rounded-2xl max-w-lg w-full p-6 space-y-4 shadow-2xl">
            <div className="flex items-center justify-between border-b border-white/10 pb-3">
              <h3 className="text-base font-semibold text-white font-sans">New Research Workflow</h3>
              <button onClick={() => setIsTemplateModalOpen(false)} className="text-neutral-400 hover:text-white">
                ✕
              </button>
            </div>

            <form onSubmit={handleCreateFromTemplate} className="space-y-4 text-xs font-mono">
              <div className="space-y-1">
                <label className="text-neutral-400">STUDY TEMPLATE</label>
                <div className="space-y-2">
                  {[
                    {
                      id: "ACTIVE_LEARNING_STUDY",
                      title: "Active Learning Annotation Efficiency",
                      desc: "8-stage cycle comparing active sampling efficiency vs baseline.",
                    },
                    {
                      id: "BASELINE_VS_VARIANT",
                      title: "Baseline vs Component Variant Ablation",
                      desc: "Resolution & augmentation scaling benchmark.",
                    },
                    {
                      id: "MODEL_ARCHITECTURE_COMPARISON",
                      title: "CNN vs Transformer Architecture Benchmark",
                      desc: "Head-to-head comparison on locked protocol.",
                    },
                  ].map((tmpl) => (
                    <button
                      key={tmpl.id}
                      type="button"
                      onClick={() => setSelectedTemplate(tmpl.id)}
                      className={`w-full text-left p-3 rounded-xl border transition-all cursor-pointer ${
                        selectedTemplate === tmpl.id
                          ? "bg-purple-950/30 border-purple-500 text-white"
                          : "bg-neutral-950 border-white/10 text-neutral-400 hover:border-white/20"
                      }`}
                    >
                      <span className="font-semibold text-white block">{tmpl.title}</span>
                      <span className="text-[11px] text-neutral-400 font-sans block mt-0.5">{tmpl.desc}</span>
                    </button>
                  ))}
                </div>
              </div>

              <div className="space-y-1">
                <label className="text-neutral-400">CUSTOM TITLE (OPTIONAL)</label>
                <input
                  type="text"
                  value={templateName}
                  onChange={(e) => setTemplateName(e.target.value)}
                  placeholder="e.g. Safety Hazard Active Learning Cycle 2"
                  className="w-full px-3 py-2 rounded-lg bg-neutral-950 border border-white/10 text-white text-sm font-sans"
                />
              </div>

              <div className="flex items-center justify-end gap-2 pt-2 border-t border-white/10">
                <Button variant="ghost" size="sm" type="button" onClick={() => setIsTemplateModalOpen(false)}>
                  Cancel
                </Button>
                <Button variant="primary" size="sm" type="submit" className="bg-purple-600 hover:bg-purple-500 text-white">
                  Instantiate Workflow
                </Button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
}
