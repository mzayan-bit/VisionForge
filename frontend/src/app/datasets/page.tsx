"use client";

import React, { useState, useEffect } from "react";
import {
  Database,
  Sliders,
  CheckCircle2,
  AlertTriangle,
  FileSpreadsheet,
  Download,
  Play,
  RotateCcw,
  Sparkles,
  Layers,
  ShieldAlert,
  Clock,
  ExternalLink,
} from "lucide-react";
import { PageHeader } from "@/components/ui/PageHeader";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";

interface ValidationIssue {
  sample_id: string;
  issue_type: string;
  message: string;
  severity: "warning" | "error";
}

interface ValidationReport {
  status: string;
  total_samples: number;
  valid_samples: number;
  corrupted_samples_count: number;
  missing_embeddings_count: number;
  issues: ValidationIssue[];
}

interface LeakageFinding {
  group_id: string;
  leakage_type: string;
  sample_ids: string[];
  similarity_score: number;
}

interface SplitStats {
  split_name: string;
  count: number;
  ratio: number;
  format_distribution: Record<string, number>;
  category_distribution: Record<string, number>;
}

interface PreparationRun {
  preparation_id: string;
  dataset_id: string;
  dataset_version: string;
  status: string;
  created_at: string;
  completed_at?: string;
  split_config: {
    train_ratio: number;
    val_ratio: number;
    test_ratio: number;
    random_seed: number;
    strategy: string;
  };
  validation_report?: ValidationReport;
  leakage_findings?: LeakageFinding[];
  split_stats?: Record<string, SplitStats>;
  manifest_path?: string;
  error_message?: string;
}

export default function DatasetPreparationPage() {
  const [datasetId, setDatasetId] = useState("safety_dataset_v2");
  const [datasetVersion, setDatasetVersion] = useState("v2.1");
  const [trainRatio, setTrainRatio] = useState(70);
  const [valRatio, setValRatio] = useState(15);
  const [testRatio, setTestRatio] = useState(15);
  const [seed, setSeed] = useState(42);
  const [strategy, setStrategy] = useState("random");

  const [isRunning, setIsRunning] = useState(false);
  const [currentRun, setCurrentRun] = useState<PreparationRun | null>(null);
  const [history, setHistory] = useState<PreparationRun[]>([]);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  useEffect(() => {
    fetchHistory();
  }, []);

  const fetchHistory = async () => {
    try {
      const res = await fetch("/api/v1/datasets/prepare/history");
      if (res.ok) {
        const payload = await res.json();
        if (payload.data) {
          setHistory(payload.data);
          if (payload.data.length > 0 && !currentRun) {
            setCurrentRun(payload.data[0]);
          }
        }
      }
    } catch {
      // Backend server may be offline during initial client mounting
    }
  };

  const handleRatioChange = (type: "train" | "val" | "test", value: number) => {
    if (type === "train") {
      setTrainRatio(value);
      const remaining = 100 - value;
      setValRatio(Math.round(remaining / 2));
      setTestRatio(remaining - Math.round(remaining / 2));
    } else if (type === "val") {
      setValRatio(value);
      setTestRatio(Math.max(0, 100 - trainRatio - value));
    } else {
      setTestRatio(value);
      setValRatio(Math.max(0, 100 - trainRatio - value));
    }
  };

  const handlePrepareDataset = async () => {
    setIsRunning(true);
    setErrorMessage(null);

    const payload = {
      dataset_id: datasetId,
      dataset_version: datasetVersion,
      train_ratio: trainRatio / 100,
      val_ratio: valRatio / 100,
      test_ratio: testRatio / 100,
      random_seed: Number(seed),
      strategy: strategy,
    };

    try {
      const res = await fetch("/api/v1/datasets/prepare", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      const result = await res.json();

      if (res.ok && result.data) {
        setCurrentRun(result.data);
        fetchHistory();
      } else {
        setErrorMessage(result.detail || result.message || "Failed to execute dataset preparation.");
      }
    } catch (err: any) {
      setErrorMessage(err.message || "Network error communicating with backend.");
    } finally {
      setIsRunning(false);
    }
  };

  const handleExportManifest = (prepId: string, format: "json" | "csv") => {
    window.open(`/api/v1/datasets/prepare/${prepId}/manifest?format=${format}`, "_blank");
  };

  const totalRatio = trainRatio + valRatio + testRatio;
  const isRatioValid = Math.abs(totalRatio - 100) <= 1;

  return (
    <div className="space-y-6 max-w-7xl mx-auto pb-12">
      <PageHeader
        title="Dataset Preparation Pipeline"
        description="Convert raw visual memory indices into reproducible, leakage-free, training-ready dataset manifests."
        breadcrumbs={["VisionForge", "Datasets", "Preparation Pipeline"]}
        actions={
          <div className="flex gap-2">
            <Button variant="secondary" icon={<RotateCcw className="w-4 h-4" />} onClick={fetchHistory}>
              Refresh Run History
            </Button>
          </div>
        }
      />

      {/* Grid Layout: Config vs Run Details */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left Panel: Configuration (5 cols) */}
        <div className="lg:col-span-5 space-y-6">
          <Card className="bg-slate-900 border-slate-800 shadow-xl">
            <CardHeader className="border-b border-slate-800 pb-4">
              <CardTitle className="text-lg font-semibold text-slate-100 flex items-center gap-2">
                <Sliders className="w-5 h-5 text-indigo-400" />
                Preparation Configuration
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-5 pt-4">
              {/* Dataset Metadata */}
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-xs font-medium text-slate-400 mb-1">Dataset ID</label>
                  <input
                    type="text"
                    value={datasetId}
                    onChange={(e) => setDatasetId(e.target.value)}
                    className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-sm text-slate-200 focus:outline-none focus:border-indigo-500"
                  />
                </div>
                <div>
                  <label className="block text-xs font-medium text-slate-400 mb-1">Version</label>
                  <input
                    type="text"
                    value={datasetVersion}
                    onChange={(e) => setDatasetVersion(e.target.value)}
                    className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-sm text-slate-200 focus:outline-none focus:border-indigo-500"
                  />
                </div>
              </div>

              {/* Split Strategy */}
              <div>
                <label className="block text-xs font-medium text-slate-400 mb-1">Split Strategy</label>
                <select
                  value={strategy}
                  onChange={(e) => setStrategy(e.target.value)}
                  className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-sm text-slate-200 focus:outline-none focus:border-indigo-500"
                >
                  <option value="random">Random Seed Split (Standard)</option>
                  <option value="stratified">Stratified Label Split (Preserve Class Ratio)</option>
                  <option value="group_aware">Group-Aware Split (Keep Groups Intact)</option>
                </select>
              </div>

              {/* Random Seed */}
              <div>
                <label className="block text-xs font-medium text-slate-400 mb-1">Random Seed (Reproducibility)</label>
                <input
                  type="number"
                  value={seed}
                  onChange={(e) => setSeed(Number(e.target.value))}
                  className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-sm text-slate-200 font-mono focus:outline-none focus:border-indigo-500"
                />
              </div>

              {/* Ratio Sliders */}
              <div className="space-y-4 pt-2 border-t border-slate-800/80">
                <div className="flex justify-between text-xs font-medium">
                  <span className="text-slate-400">Partition Ratios</span>
                  <span className={isRatioValid ? "text-emerald-400" : "text-rose-400 font-bold"}>
                    Total: {totalRatio}% {isRatioValid ? "✓" : "(!= 100%)"}
                  </span>
                </div>

                {/* Train */}
                <div className="space-y-1">
                  <div className="flex justify-between text-xs text-indigo-300">
                    <span>Train Partition</span>
                    <span className="font-mono">{trainRatio}%</span>
                  </div>
                  <input
                    type="range"
                    min="10"
                    max="90"
                    value={trainRatio}
                    onChange={(e) => handleRatioChange("train", Number(e.target.value))}
                    className="w-full h-1.5 bg-slate-950 rounded-lg appearance-none cursor-pointer accent-indigo-500"
                  />
                </div>

                {/* Validation */}
                <div className="space-y-1">
                  <div className="flex justify-between text-xs text-emerald-300">
                    <span>Validation Partition</span>
                    <span className="font-mono">{valRatio}%</span>
                  </div>
                  <input
                    type="range"
                    min="5"
                    max="40"
                    value={valRatio}
                    onChange={(e) => handleRatioChange("val", Number(e.target.value))}
                    className="w-full h-1.5 bg-slate-950 rounded-lg appearance-none cursor-pointer accent-emerald-500"
                  />
                </div>

                {/* Test */}
                <div className="space-y-1">
                  <div className="flex justify-between text-xs text-amber-300">
                    <span>Test Partition</span>
                    <span className="font-mono">{testRatio}%</span>
                  </div>
                  <input
                    type="range"
                    min="5"
                    max="40"
                    value={testRatio}
                    onChange={(e) => handleRatioChange("test", Number(e.target.value))}
                    className="w-full h-1.5 bg-slate-950 rounded-lg appearance-none cursor-pointer accent-amber-500"
                  />
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
                icon={isRunning ? <Sparkles className="w-4 h-4 animate-spin" /> : <Play className="w-4 h-4" />}
                disabled={isRunning || !isRatioValid}
                onClick={handlePrepareDataset}
              >
                {isRunning ? "Executing Pipeline..." : "Validate & Prepare Dataset"}
              </Button>
            </CardContent>
          </Card>
        </div>

        {/* Right Panel: Run Details & Validation Report (7 cols) */}
        <div className="lg:col-span-7 space-y-6">
          {currentRun ? (
            <>
              {/* Status Header Banner */}
              <Card className="bg-slate-900 border-slate-800 shadow-xl overflow-hidden">
                <div className="p-5 border-b border-slate-800 flex justify-between items-center bg-slate-950/50">
                  <div>
                    <div className="flex items-center gap-2">
                      <span className="text-sm font-semibold text-slate-200">
                        {currentRun.dataset_id} ({currentRun.dataset_version})
                      </span>
                      <span className="px-2 py-0.5 rounded text-xs font-mono bg-indigo-500/10 text-indigo-300 border border-indigo-500/20">
                        {currentRun.preparation_id}
                      </span>
                    </div>
                    <p className="text-xs text-slate-500 mt-1 flex items-center gap-2">
                      <Clock className="w-3.5 h-3.5" />
                      Created at {new Date(currentRun.created_at).toLocaleString()}
                    </p>
                  </div>
                  <div className="flex items-center gap-3">
                    <span
                      className={`px-3 py-1 rounded-full text-xs font-bold ${
                        currentRun.status === "COMPLETED"
                          ? "bg-emerald-500/10 text-emerald-400 border border-emerald-500/30"
                          : currentRun.status === "FAILED"
                          ? "bg-rose-500/10 text-rose-400 border border-rose-500/30"
                          : "bg-amber-500/10 text-amber-400 border border-amber-500/30 animate-pulse"
                      }`}
                    >
                      {currentRun.status}
                    </span>
                    {currentRun.status === "COMPLETED" && (
                      <div className="flex gap-2">
                        <Button
                          variant="secondary"
                          size="sm"
                          icon={<Download className="w-3.5 h-3.5" />}
                          onClick={() => handleExportManifest(currentRun.preparation_id, "json")}
                        >
                          JSON
                        </Button>
                        <Button
                          variant="secondary"
                          size="sm"
                          icon={<FileSpreadsheet className="w-3.5 h-3.5" />}
                          onClick={() => handleExportManifest(currentRun.preparation_id, "csv")}
                        >
                          CSV
                        </Button>
                      </div>
                    )}
                  </div>
                </div>

                <CardContent className="p-6 space-y-6">
                  {/* Split Preview Visual Bar */}
                  {currentRun.split_stats && (
                    <div className="space-y-2">
                      <div className="flex justify-between text-xs font-medium text-slate-300">
                        <span>Split Partition Preview</span>
                        <span className="font-mono text-slate-400">
                          Seed: {currentRun.split_config.random_seed} | Strategy: {currentRun.split_config.strategy}
                        </span>
                      </div>
                      <div className="h-4 w-full bg-slate-950 rounded-lg overflow-hidden flex border border-slate-800">
                        <div
                          style={{
                            width: `${(currentRun.split_stats.train?.ratio || 0.7) * 100}%`,
                          }}
                          className="bg-indigo-500 h-full flex items-center justify-center text-[10px] font-bold text-white"
                          title={`Train: ${currentRun.split_stats.train?.count} samples`}
                        >
                          Train (
                          {Math.round((currentRun.split_stats.train?.ratio || 0.7) * 100)}
                          %)
                        </div>
                        <div
                          style={{
                            width: `${(currentRun.split_stats.val?.ratio || 0.15) * 100}%`,
                          }}
                          className="bg-emerald-500 h-full flex items-center justify-center text-[10px] font-bold text-slate-950"
                          title={`Val: ${currentRun.split_stats.val?.count} samples`}
                        >
                          Val (
                          {Math.round((currentRun.split_stats.val?.ratio || 0.15) * 100)}
                          %)
                        </div>
                        <div
                          style={{
                            width: `${(currentRun.split_stats.test?.ratio || 0.15) * 100}%`,
                          }}
                          className="bg-amber-500 h-full flex items-center justify-center text-[10px] font-bold text-slate-950"
                          title={`Test: ${currentRun.split_stats.test?.count} samples`}
                        >
                          Test (
                          {Math.round((currentRun.split_stats.test?.ratio || 0.15) * 100)}
                          %)
                        </div>
                      </div>

                      {/* Numeric Breakdown Cards */}
                      <div className="grid grid-cols-3 gap-3 pt-2">
                        <div className="p-3 bg-indigo-500/10 border border-indigo-500/20 rounded-lg">
                          <p className="text-[11px] font-semibold text-indigo-400">TRAIN</p>
                          <p className="text-lg font-bold text-slate-100 font-mono">
                            {currentRun.split_stats.train?.count || 0}
                          </p>
                          <p className="text-[10px] text-slate-500">samples</p>
                        </div>
                        <div className="p-3 bg-emerald-500/10 border border-emerald-500/20 rounded-lg">
                          <p className="text-[11px] font-semibold text-emerald-400">VALIDATION</p>
                          <p className="text-lg font-bold text-slate-100 font-mono">
                            {currentRun.split_stats.validation?.count || 0}
                          </p>
                          <p className="text-[10px] text-slate-500">samples</p>
                        </div>
                        <div className="p-3 bg-amber-500/10 border border-amber-500/20 rounded-lg">
                          <p className="text-[11px] font-semibold text-amber-400">TEST</p>
                          <p className="text-lg font-bold text-slate-100 font-mono">
                            {currentRun.split_stats.test?.count || 0}
                          </p>
                          <p className="text-[10px] text-slate-500">samples</p>
                        </div>
                      </div>
                    </div>
                  )}

                  {/* Pre-split Validation Report */}
                  {currentRun.validation_report && (
                    <div className="space-y-3 pt-4 border-t border-slate-800">
                      <h4 className="text-sm font-semibold text-slate-300 flex items-center gap-2">
                        <CheckCircle2 className="w-4 h-4 text-emerald-400" />
                        Pre-Split Validation Report
                      </h4>

                      <div className="grid grid-cols-3 gap-3 text-xs">
                        <div className="p-2.5 bg-slate-950 border border-slate-800 rounded-lg">
                          <span className="text-slate-400">Total Checked:</span>
                          <span className="float-right font-mono font-bold text-slate-200">
                            {currentRun.validation_report.total_samples}
                          </span>
                        </div>
                        <div className="p-2.5 bg-slate-950 border border-slate-800 rounded-lg">
                          <span className="text-slate-400">Valid Samples:</span>
                          <span className="float-right font-mono font-bold text-emerald-400">
                            {currentRun.validation_report.valid_samples}
                          </span>
                        </div>
                        <div className="p-2.5 bg-slate-950 border border-slate-800 rounded-lg">
                          <span className="text-slate-400">Corrupted:</span>
                          <span className="float-right font-mono font-bold text-rose-400">
                            {currentRun.validation_report.corrupted_samples_count}
                          </span>
                        </div>
                      </div>
                    </div>
                  )}

                  {/* Data Leakage Findings Panel */}
                  <div className="space-y-3 pt-4 border-t border-slate-800">
                    <h4 className="text-sm font-semibold text-slate-300 flex items-center gap-2">
                      <ShieldAlert className="w-4 h-4 text-amber-400" />
                      Data Leakage Prevention Analysis
                    </h4>

                    {currentRun.leakage_findings && currentRun.leakage_findings.length > 0 ? (
                      <div className="space-y-2">
                        {currentRun.leakage_findings.map((finding) => (
                          <div
                            key={finding.group_id}
                            className="p-3 bg-amber-500/10 border border-amber-500/20 rounded-lg flex justify-between items-center text-xs"
                          >
                            <div className="space-y-0.5">
                              <div className="flex items-center gap-2">
                                <span className="font-bold text-amber-300">{finding.leakage_type}</span>
                                <span className="font-mono text-slate-400">({finding.group_id})</span>
                              </div>
                              <p className="text-[11px] text-slate-400">
                                {finding.sample_ids.length} samples grouped together into same partition.
                              </p>
                            </div>
                            <span className="font-mono font-bold text-amber-400 bg-amber-500/20 px-2 py-0.5 rounded">
                              Score: {finding.similarity_score}
                            </span>
                          </div>
                        ))}
                      </div>
                    ) : (
                      <div className="p-3 bg-emerald-500/10 border border-emerald-500/20 rounded-lg text-xs text-emerald-400 flex items-center gap-2">
                        <CheckCircle2 className="w-4 h-4" />
                        No exact or near-duplicate data leakage detected across splits.
                      </div>
                    )}
                  </div>
                </CardContent>
              </Card>
            </>
          ) : (
            <Card className="bg-slate-900 border-slate-800 shadow-xl p-12 text-center">
              <Database className="w-12 h-12 text-indigo-400 mx-auto mb-4 opacity-50" />
              <h3 className="text-lg font-bold text-slate-200">No Active Preparation Run</h3>
              <p className="text-xs text-slate-500 max-w-sm mx-auto mt-2">
                Configure your split ratios, random seed, and split strategy on the left, then click &quot;Validate &amp;
                Prepare Dataset&quot;.
              </p>
            </Card>
          )}
        </div>
      </div>

      {/* Bottom Section: Preparation History Log Table */}
      <Card className="bg-slate-900 border-slate-800 shadow-xl">
        <CardHeader className="border-b border-slate-800 pb-4 flex flex-row items-center justify-between">
          <CardTitle className="text-md font-semibold text-slate-200 flex items-center gap-2">
            <Clock className="w-4 h-4 text-indigo-400" />
            Preparation Run History &amp; Reproducibility Audit Log
          </CardTitle>
        </CardHeader>
        <CardContent className="p-0 overflow-x-auto">
          {history.length > 0 ? (
            <table className="w-full text-left text-xs">
              <thead className="bg-slate-950/80 text-slate-400 border-b border-slate-800 font-mono">
                <tr>
                  <th className="p-3 pl-6">Prep ID</th>
                  <th className="p-3">Dataset / Version</th>
                  <th className="p-3">Strategy</th>
                  <th className="p-3">Seed</th>
                  <th className="p-3">Ratios (T / V / T)</th>
                  <th className="p-3">Status</th>
                  <th className="p-3">Timestamp</th>
                  <th className="p-3 pr-6 text-right">Actions</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/60 text-slate-300">
                {history.map((run) => (
                  <tr key={run.preparation_id} className="hover:bg-slate-800/30 transition-colors">
                    <td className="p-3 pl-6 font-mono text-indigo-300 font-bold">{run.preparation_id}</td>
                    <td className="p-3">
                      {run.dataset_id} ({run.dataset_version})
                    </td>
                    <td className="p-3 capitalize">{run.split_config.strategy}</td>
                    <td className="p-3 font-mono">{run.split_config.random_seed}</td>
                    <td className="p-3 font-mono text-slate-400">
                      {Math.round(run.split_config.train_ratio * 100)} /{" "}
                      {Math.round(run.split_config.val_ratio * 100)} /{" "}
                      {Math.round(run.split_config.test_ratio * 100)}
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
                    <td className="p-3 text-slate-500">{new Date(run.created_at).toLocaleString()}</td>
                    <td className="p-3 pr-6 text-right">
                      <div className="flex justify-end gap-2">
                        <Button
                          variant="secondary"
                          size="sm"
                          className="h-7 text-[11px]"
                          onClick={() => setCurrentRun(run)}
                        >
                          Inspect
                        </Button>
                        <Button
                          variant="secondary"
                          size="sm"
                          className="h-7 text-[11px]"
                          icon={<Download className="w-3 h-3" />}
                          onClick={() => handleExportManifest(run.preparation_id, "json")}
                        >
                          JSON
                        </Button>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          ) : (
            <div className="p-8 text-center text-xs text-slate-500">
              No historical preparation runs logged yet.
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
