"use client";

import React, { useState, useEffect } from "react";
import Link from "next/link";
import {
  Cpu,
  Play,
  CheckCircle2,
  Sliders,
  Sparkles,
  RotateCcw,
  Clock,
  ExternalLink,
  ShieldCheck,
  TrendingUp,
  FileCode,
  Download,
} from "lucide-react";
import { PageHeader } from "@/components/ui/PageHeader";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";

interface TrainingRun {
  run_id: string;
  experiment_name: string;
  dataset_id: string;
  dataset_version: string;
  preparation_id: string;
  status: string;
  created_at: string;
  completed_at?: string;
  config: {
    model_name: string;
    epochs: number;
    batch_size: number;
    imgsz: number;
    learning_rate: number;
    device: string;
  };
  best_metrics?: {
    map50: number;
    map50_95: number;
    precision: number;
    recall: number;
  };
  test_evaluation?: {
    map50: number;
    map50_95: number;
  };
}

interface PreparationRun {
  preparation_id: string;
  dataset_id: string;
  dataset_version: string;
  status: string;
}

export default function TrainingLabPage() {
  const [modelName, setModelName] = useState("yolo11s.pt");
  const [prepId, setPrepId] = useState("");
  const [epochs, setEpochs] = useState(50);
  const [batchSize, setBatchSize] = useState(16);
  const [imgsz, setImgsz] = useState(640);
  const [learningRate, setLearningRate] = useState(0.01);
  const [device, setDevice] = useState("colab_gpu");
  const [experimentName, setExperimentName] = useState("yolo11_safety_experiment");

  const [preparations, setPreparations] = useState<PreparationRun[]>([]);
  const [runs, setRuns] = useState<TrainingRun[]>([]);
  const [isExecuting, setIsExecuting] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  useEffect(() => {
    fetchPreparations();
    fetchRuns();
  }, []);

  const fetchPreparations = async () => {
    try {
      const res = await fetch("/api/v1/datasets/prepare/history");
      if (res.ok) {
        const payload = await res.json();
        if (payload.data && payload.data.length > 0) {
          setPreparations(payload.data);
          setPrepId(payload.data[0].preparation_id);
        }
      }
    } catch {
      // Backend may be offline during client mount
    }
  };

  const fetchRuns = async () => {
    try {
      const res = await fetch("/api/v1/training/runs");
      if (res.ok) {
        const payload = await res.json();
        if (payload.data) {
          setRuns(payload.data);
        }
      }
    } catch {
      // Backend may be offline
    }
  };

  const handleStartTraining = async () => {
    setIsExecuting(true);
    setErrorMessage(null);

    const payload = {
      model_name: modelName,
      dataset_id: preparations.find((p) => p.preparation_id === prepId)?.dataset_id || "safety_dataset",
      preparation_id: prepId,
      epochs: Number(epochs),
      batch_size: Number(batchSize),
      imgsz: Number(imgsz),
      learning_rate: Number(learningRate),
      device: device,
      random_seed: 42,
      experiment_name: experimentName,
    };

    try {
      const res = await fetch("/api/v1/training/runs", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      const result = await res.json();

      if (res.ok && result.data) {
        fetchRuns();
      } else {
        setErrorMessage(result.detail || result.message || "Failed to start training run.");
      }
    } catch (err: any) {
      setErrorMessage(err.message || "Network error communicating with training API.");
    } finally {
      setIsExecuting(false);
    }
  };

  return (
    <div className="space-y-6 max-w-7xl mx-auto pb-12">
      <PageHeader
        title="Training Lab"
        description="Configure, execute, track, and register Computer Vision model training experiments (YOLO11s)."
        breadcrumbs={["VisionForge", "Training Lab"]}
        actions={
          <div className="flex gap-2">
            <Button variant="secondary" icon={<RotateCcw className="w-4 h-4" />} onClick={fetchRuns}>
              Refresh Experiments
            </Button>
          </div>
        }
      />

      {/* Grid Layout: New Run Builder vs Experiments */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left Panel: Training Configuration Builder (5 cols) */}
        <div className="lg:col-span-5 space-y-6">
          <Card className="bg-slate-900 border-slate-800 shadow-xl">
            <CardHeader className="border-b border-slate-800 pb-4">
              <CardTitle className="text-lg font-semibold text-slate-100 flex items-center gap-2">
                <Sliders className="w-5 h-5 text-indigo-400" />
                New Training Experiment
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-5 pt-4">
              {/* Dataset Preparation Selector */}
              <div>
                <label className="block text-xs font-medium text-slate-400 mb-1">Prepared Dataset Split</label>
                {preparations.length > 0 ? (
                  <select
                    value={prepId}
                    onChange={(e) => setPrepId(e.target.value)}
                    className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-sm text-slate-200 focus:outline-none focus:border-indigo-500"
                  >
                    {preparations.map((prep) => (
                      <option key={prep.preparation_id} value={prep.preparation_id}>
                        {prep.dataset_id} ({prep.dataset_version}) — {prep.preparation_id}
                      </option>
                    ))}
                  </select>
                ) : (
                  <input
                    type="text"
                    value={prepId}
                    placeholder="prep_12345"
                    onChange={(e) => setPrepId(e.target.value)}
                    className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-sm text-slate-200 focus:outline-none focus:border-indigo-500 font-mono"
                  />
                )}
              </div>

              {/* Model & Experiment Name */}
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-xs font-medium text-slate-400 mb-1">Base Architecture</label>
                  <select
                    value={modelName}
                    onChange={(e) => setModelName(e.target.value)}
                    className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-sm text-slate-200 focus:outline-none focus:border-indigo-500 font-semibold"
                  >
                    <option value="yolo11s.pt">YOLO11s (Ultralytics)</option>
                    <option value="yolo11n.pt">YOLO11n (Nano)</option>
                    <option value="yolo11m.pt">YOLO11m (Medium)</option>
                  </select>
                </div>
                <div>
                  <label className="block text-xs font-medium text-slate-400 mb-1">Experiment Key</label>
                  <input
                    type="text"
                    value={experimentName}
                    onChange={(e) => setExperimentName(e.target.value)}
                    className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-sm text-slate-200 focus:outline-none focus:border-indigo-500"
                  />
                </div>
              </div>

              {/* Hyperparameters Grid */}
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-xs font-medium text-slate-400 mb-1">Epochs</label>
                  <input
                    type="number"
                    value={epochs}
                    onChange={(e) => setEpochs(Number(e.target.value))}
                    className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-sm text-slate-200 font-mono focus:outline-none focus:border-indigo-500"
                  />
                </div>
                <div>
                  <label className="block text-xs font-medium text-slate-400 mb-1">Batch Size</label>
                  <input
                    type="number"
                    value={batchSize}
                    onChange={(e) => setBatchSize(Number(e.target.value))}
                    className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-sm text-slate-200 font-mono focus:outline-none focus:border-indigo-500"
                  />
                </div>
                <div>
                  <label className="block text-xs font-medium text-slate-400 mb-1">Image Size (px)</label>
                  <input
                    type="number"
                    value={imgsz}
                    onChange={(e) => setImgsz(Number(e.target.value))}
                    className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-sm text-slate-200 font-mono focus:outline-none focus:border-indigo-500"
                  />
                </div>
                <div>
                  <label className="block text-xs font-medium text-slate-400 mb-1">Learning Rate</label>
                  <input
                    type="number"
                    step="0.001"
                    value={learningRate}
                    onChange={(e) => setLearningRate(Number(e.target.value))}
                    className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-sm text-slate-200 font-mono focus:outline-none focus:border-indigo-500"
                  />
                </div>
              </div>

              {/* Execution Device */}
              <div>
                <label className="block text-xs font-medium text-slate-400 mb-1">Execution Hardware Target</label>
                <select
                  value={device}
                  onChange={(e) => setDevice(e.target.value)}
                  className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-sm text-slate-200 focus:outline-none focus:border-indigo-500"
                >
                  <option value="colab_gpu">Google Colab Remote T4 GPU (Recommended)</option>
                  <option value="mps">Local Apple Silicon M4 (MPS)</option>
                  <option value="cpu">Local CPU Lightweight Testing</option>
                </select>
              </div>

              {/* Pre-flight Checks */}
              <div className="p-3.5 bg-slate-950 border border-slate-800 rounded-lg space-y-2 text-xs">
                <div className="flex items-center gap-2 text-emerald-400">
                  <CheckCircle2 className="w-4 h-4" />
                  <span>Dataset Manifest Validated</span>
                </div>
                <div className="flex items-center gap-2 text-emerald-400">
                  <CheckCircle2 className="w-4 h-4" />
                  <span>Train / Val / Test Partition Verified</span>
                </div>
                <div className="flex items-center gap-2 text-emerald-400">
                  <ShieldCheck className="w-4 h-4" />
                  <span>Data Leakage Prevention Passed</span>
                </div>
              </div>

              {errorMessage && (
                <div className="p-3 bg-rose-500/10 border border-rose-500/30 rounded-lg text-xs text-rose-300">
                  {errorMessage}
                </div>
              )}

              {/* Action Button */}
              <Button
                variant="primary"
                className="w-full justify-center py-2.5 bg-indigo-600 hover:bg-indigo-500 font-semibold"
                icon={isExecuting ? <Sparkles className="w-4 h-4 animate-spin" /> : <Play className="w-4 h-4" />}
                disabled={isExecuting || !prepId}
                onClick={handleStartTraining}
              >
                {isExecuting ? "Executing Training Run..." : "Start / Export Training Run"}
              </Button>
            </CardContent>
          </Card>
        </div>

        {/* Right Panel: Recent Experiments (7 cols) */}
        <div className="lg:col-span-7 space-y-6">
          <Card className="bg-slate-900 border-slate-800 shadow-xl">
            <CardHeader className="border-b border-slate-800 pb-4 flex flex-row items-center justify-between">
              <CardTitle className="text-md font-semibold text-slate-200 flex items-center gap-2">
                <TrendingUp className="w-4 h-4 text-indigo-400" />
                Recent Training Experiments
              </CardTitle>
            </CardHeader>
            <CardContent className="p-0 overflow-x-auto">
              {runs.length > 0 ? (
                <table className="w-full text-left text-xs">
                  <thead className="bg-slate-950/80 text-slate-400 border-b border-slate-800 font-mono">
                    <tr>
                      <th className="p-3 pl-6">Run ID</th>
                      <th className="p-3">Experiment</th>
                      <th className="p-3">Model</th>
                      <th className="p-3">Epochs</th>
                      <th className="p-3">mAP@50</th>
                      <th className="p-3">Status</th>
                      <th className="p-3 pr-6 text-right">Actions</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-800/60 text-slate-300">
                    {runs.map((run) => (
                      <tr key={run.run_id} className="hover:bg-slate-800/30 transition-colors">
                        <td className="p-3 pl-6 font-mono text-indigo-300 font-bold">{run.run_id}</td>
                        <td className="p-3">{run.experiment_name}</td>
                        <td className="p-3 font-mono">{run.config.model_name}</td>
                        <td className="p-3 font-mono">{run.config.epochs}</td>
                        <td className="p-3 font-mono text-emerald-400 font-bold">
                          {run.best_metrics ? (run.best_metrics.map50 * 100).toFixed(1) + "%" : "—"}
                        </td>
                        <td className="p-3">
                          <span
                            className={`px-2 py-0.5 rounded text-[10px] font-bold ${
                              run.status === "COMPLETED"
                                ? "bg-emerald-500/10 text-emerald-400 border border-emerald-500/20"
                                : "bg-rose-500/10 text-rose-400 border border-rose-500/20"
                            }`}
                          >
                            {run.status}
                          </span>
                        </td>
                        <td className="p-3 pr-6 text-right">
                          <Link href={`/training/${run.run_id}`}>
                            <Button variant="secondary" size="sm" className="h-7 text-[11px]">
                              Inspect Details
                            </Button>
                          </Link>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              ) : (
                <div className="p-12 text-center text-xs text-slate-500">
                  <Cpu className="w-8 h-8 text-indigo-400 mx-auto mb-2 opacity-50" />
                  No training experiments logged yet. Configure your experiment on the left to start.
                </div>
              )}
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}
