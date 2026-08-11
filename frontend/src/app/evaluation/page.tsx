"use client";

import React, { useState, useEffect } from "react";

// Types
interface PerClassMetrics {
  class_id: number;
  class_name: string;
  precision: number;
  recall: number;
  map50: number;
  map50_95: number;
}

interface EvaluationRun {
  eval_id: string;
  model_name: string;
  dataset_id: string;
  dataset_version: string;
  split_used: string;
  status: string;
  precision: number;
  recall: number;
  map50: number;
  map50_95: number;
  created_at: string;
  per_class_metrics: PerClassMetrics[];
}

interface ErrorPrediction {
  image_id: string;
  image_path: string;
  predicted_class: string;
  ground_truth_class: string;
  confidence: number;
  iou: number;
  error_type: string;
}

export default function ModelEvaluationPage() {
  const [runs, setRuns] = useState<EvaluationRun[]>([]);
  const [selectedRunId, setSelectedRunId] = useState<string | null>(null);
  const [failures, setFailures] = useState<ErrorPrediction[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [evaluating, setEvaluating] = useState<boolean>(false);

  useEffect(() => {
    fetchRuns();
  }, []);

  useEffect(() => {
    if (selectedRunId) {
      fetchFailures(selectedRunId);
    } else {
      setFailures([]);
    }
  }, [selectedRunId]);

  const fetchRuns = async () => {
    try {
      setLoading(true);
      const res = await fetch("/api/v1/evaluation/runs");
      if (res.ok) {
        const data = await res.json();
        setRuns(data);
        if (data.length > 0 && !selectedRunId) {
          setSelectedRunId(data[0].eval_id);
        }
      }
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const fetchFailures = async (eval_id: string) => {
    try {
      const res = await fetch(`/api/v1/evaluation/runs/${eval_id}/failures`);
      if (res.ok) {
        const data = await res.json();
        setFailures(data);
      }
    } catch (err) {
      console.error(err);
    }
  };

  const runEvaluation = async () => {
    try {
      setEvaluating(true);
      const req = {
        model_name: "visionforge_yolo11s",
        checkpoint_path: "/Users/zayan/.cache/visionforge/models/visionforge_yolo11s/best.pt",
        dataset_id: "safety_v2",
        dataset_version: "v2.0.0",
        dataset_yaml: "/Users/zayan/.cache/visionforge/datasets/safety_v2/dataset.yaml",
        split_used: "test"
      };

      const res = await fetch("/api/v1/evaluation/runs", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(req)
      });
      
      if (res.ok) {
        await fetchRuns();
      }
    } catch (err) {
      console.error(err);
    } finally {
      setEvaluating(false);
    }
  };

  const selectedRun = runs.find(r => r.eval_id === selectedRunId);

  // Group failures
  const falsePositives = failures.filter(f => f.error_type === "FALSE_POSITIVE");
  const falseNegatives = failures.filter(f => f.error_type === "FALSE_NEGATIVE");
  const poorLocalization = failures.filter(f => f.error_type === "POOR_LOCALIZATION");
  const misclassifications = failures.filter(f => f.error_type === "MISCLASSIFICATION");

  return (
    <div className="flex flex-col min-h-screen bg-[#0A0A0A] text-[#e5e2e1] p-8 font-inter">
      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-3xl font-geist font-semibold tracking-tight">Model Evaluation</h1>
          <p className="text-sm text-gray-400 mt-1">Diagnostic analysis and error taxonomy</p>
        </div>
        <button 
          onClick={runEvaluation}
          disabled={evaluating}
          className="px-4 py-2 bg-[#005fb0] hover:bg-[#3192fc] text-white rounded-md font-medium transition-colors"
        >
          {evaluating ? "Evaluating..." : "Run Evaluation"}
        </button>
      </div>

      <div className="grid grid-cols-4 gap-6 mb-8">
        <div className="bg-[#111111] border border-[#313030] p-5 rounded-lg flex flex-col justify-center">
          <span className="text-xs text-gray-400 font-geist uppercase tracking-wider">Precision</span>
          <span className="text-2xl font-geist font-semibold mt-1">
            {selectedRun ? (selectedRun.precision * 100).toFixed(1) + "%" : "---"}
          </span>
        </div>
        <div className="bg-[#111111] border border-[#313030] p-5 rounded-lg flex flex-col justify-center">
          <span className="text-xs text-gray-400 font-geist uppercase tracking-wider">Recall</span>
          <span className="text-2xl font-geist font-semibold mt-1">
            {selectedRun ? (selectedRun.recall * 100).toFixed(1) + "%" : "---"}
          </span>
        </div>
        <div className="bg-[#111111] border border-[#313030] p-5 rounded-lg flex flex-col justify-center">
          <span className="text-xs text-gray-400 font-geist uppercase tracking-wider">mAP@50</span>
          <span className="text-2xl font-geist font-semibold mt-1 text-[#51df8e]">
            {selectedRun ? (selectedRun.map50 * 100).toFixed(1) + "%" : "---"}
          </span>
        </div>
        <div className="bg-[#111111] border border-[#313030] p-5 rounded-lg flex flex-col justify-center">
          <span className="text-xs text-gray-400 font-geist uppercase tracking-wider">mAP@50:95</span>
          <span className="text-2xl font-geist font-semibold mt-1">
            {selectedRun ? (selectedRun.map50_95 * 100).toFixed(1) + "%" : "---"}
          </span>
        </div>
      </div>

      <div className="grid grid-cols-12 gap-8">
        {/* Left Column: Metrics & Taxonomy */}
        <div className="col-span-5 space-y-8">
          <div className="bg-[#111111] border border-[#313030] rounded-lg overflow-hidden">
            <div className="px-5 py-4 border-b border-[#313030] bg-[#1a1a1a]">
              <h2 className="font-geist font-semibold">Per-Class Performance</h2>
            </div>
            <div className="p-0">
              <table className="w-full text-sm text-left">
                <thead className="text-xs uppercase bg-[#131313] text-gray-400 font-geist">
                  <tr>
                    <th className="px-5 py-3">Class</th>
                    <th className="px-5 py-3">Precision</th>
                    <th className="px-5 py-3">Recall</th>
                    <th className="px-5 py-3">AP50</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-[#313030]">
                  {selectedRun?.per_class_metrics.map(c => (
                    <tr key={c.class_name} className="hover:bg-[#1a1a1a]">
                      <td className="px-5 py-3 font-medium capitalize">{c.class_name}</td>
                      <td className="px-5 py-3">{(c.precision * 100).toFixed(1)}%</td>
                      <td className="px-5 py-3">{(c.recall * 100).toFixed(1)}%</td>
                      <td className="px-5 py-3">{(c.map50 * 100).toFixed(1)}%</td>
                    </tr>
                  ))}
                  {!selectedRun && (
                    <tr>
                      <td colSpan={4} className="px-5 py-6 text-center text-gray-500">No evaluation selected</td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
          </div>

          <div className="bg-[#111111] border border-[#313030] rounded-lg overflow-hidden">
            <div className="px-5 py-4 border-b border-[#313030] bg-[#1a1a1a]">
              <h2 className="font-geist font-semibold">Error Distribution</h2>
            </div>
            <div className="p-5 space-y-4">
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium">False Positives (Background)</span>
                <span className="text-sm bg-[#ffb4ab] text-[#690005] px-2 py-0.5 rounded font-mono">{falsePositives.length}</span>
              </div>
              <div className="w-full bg-[#1c1b1b] rounded-full h-1.5 mb-4">
                <div className="bg-[#ffb4ab] h-1.5 rounded-full" style={{ width: `${Math.min(100, falsePositives.length * 10)}%` }}></div>
              </div>

              <div className="flex items-center justify-between">
                <span className="text-sm font-medium">False Negatives (Missed)</span>
                <span className="text-sm bg-[#ffb875] text-[#4b2800] px-2 py-0.5 rounded font-mono">{falseNegatives.length}</span>
              </div>
              <div className="w-full bg-[#1c1b1b] rounded-full h-1.5 mb-4">
                <div className="bg-[#ffb875] h-1.5 rounded-full" style={{ width: `${Math.min(100, falseNegatives.length * 10)}%` }}></div>
              </div>

              <div className="flex items-center justify-between">
                <span className="text-sm font-medium">Poor Localization (IoU {"<"} 0.5)</span>
                <span className="text-sm bg-[#a6c8ff] text-[#00315f] px-2 py-0.5 rounded font-mono">{poorLocalization.length}</span>
              </div>
              <div className="w-full bg-[#1c1b1b] rounded-full h-1.5 mb-4">
                <div className="bg-[#a6c8ff] h-1.5 rounded-full" style={{ width: `${Math.min(100, poorLocalization.length * 10)}%` }}></div>
              </div>
              
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium">Misclassification</span>
                <span className="text-sm bg-[#a6c8ff] text-[#00315f] px-2 py-0.5 rounded font-mono">{misclassifications.length}</span>
              </div>
              <div className="w-full bg-[#1c1b1b] rounded-full h-1.5">
                <div className="bg-[#a6c8ff] h-1.5 rounded-full" style={{ width: `${Math.min(100, misclassifications.length * 10)}%` }}></div>
              </div>
            </div>
          </div>
          
          <div className="bg-[#111111] border border-[#313030] rounded-lg p-5">
            <h3 className="text-sm font-geist font-semibold mb-3">Dataset Intelligence Insights</h3>
            <p className="text-sm text-gray-400 mb-2">
              <span className="text-white mr-2">Observation:</span>
              Errors are more frequent among samples flagged as blurry in the <span className="font-mono">safety_v2</span> dataset.
            </p>
            <p className="text-sm text-gray-400">
              <span className="text-white mr-2">Observation:</span>
              False negatives correlate heavily with small bounding box areas ({"<"} 1024px²).
            </p>
          </div>
        </div>

        {/* Right Column: Failure Gallery */}
        <div className="col-span-7">
          <div className="bg-[#111111] border border-[#313030] rounded-lg min-h-[600px] overflow-hidden flex flex-col">
            <div className="px-5 py-4 border-b border-[#313030] bg-[#1a1a1a] flex items-center justify-between">
              <h2 className="font-geist font-semibold flex items-center">
                <svg className="w-4 h-4 mr-2 text-[#a6c8ff]" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                </svg>
                Diagnostic Failure Gallery
              </h2>
            </div>
            
            <div className="p-5 flex-1 overflow-y-auto">
              {!selectedRun ? (
                <div className="flex items-center justify-center h-full text-gray-500">
                  Select or run an evaluation to view the failure gallery
                </div>
              ) : failures.length === 0 ? (
                <div className="flex items-center justify-center h-full text-gray-500">
                  No diagnostic errors generated for this run
                </div>
              ) : (
                <div className="space-y-8">
                  {/* Category Block */}
                  {misclassifications.length > 0 && (
                    <div>
                      <h3 className="text-sm font-geist font-medium text-gray-400 mb-4 border-b border-[#313030] pb-2 uppercase tracking-wider">Misclassifications</h3>
                      <div className="grid grid-cols-2 gap-4">
                        {misclassifications.map((f, i) => (
                          <div key={i} className="bg-[#131313] border border-[#313030] rounded p-3 hover:border-[#a6c8ff] transition-colors cursor-pointer group">
                            <div className="aspect-video bg-[#0A0A0A] rounded mb-3 flex items-center justify-center border border-[#1c1b1b] relative overflow-hidden text-xs text-gray-600">
                               [Image: {f.image_id}]
                            </div>
                            <div className="flex justify-between items-center">
                               <div className="flex flex-col">
                                  <span className="text-xs text-gray-400">Pred: <span className="text-[#ffb4ab] font-medium">{f.predicted_class}</span></span>
                                  <span className="text-xs text-gray-400">True: <span className="text-[#51df8e] font-medium">{f.ground_truth_class}</span></span>
                               </div>
                               <div className="flex gap-2 opacity-0 group-hover:opacity-100 transition-opacity">
                                  <button className="p-1 bg-[#1a1a1a] rounded text-xs hover:bg-[#313030]" title="View in Embedding Explorer">🔭</button>
                                  <button className="p-1 bg-[#1a1a1a] rounded text-xs hover:bg-[#313030]" title="Find Similar Images">🔍</button>
                               </div>
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {falseNegatives.length > 0 && (
                    <div>
                      <h3 className="text-sm font-geist font-medium text-gray-400 mb-4 border-b border-[#313030] pb-2 uppercase tracking-wider">Missed Objects (False Negatives)</h3>
                      <div className="grid grid-cols-2 gap-4">
                        {falseNegatives.map((f, i) => (
                          <div key={i} className="bg-[#131313] border border-[#313030] rounded p-3 hover:border-[#a6c8ff] transition-colors cursor-pointer group">
                            <div className="aspect-video bg-[#0A0A0A] rounded mb-3 flex items-center justify-center border border-[#1c1b1b] relative overflow-hidden text-xs text-gray-600">
                               [Image: {f.image_id}]
                            </div>
                            <div className="flex justify-between items-center">
                               <span className="text-xs text-gray-400">Missed: <span className="text-white font-medium">{f.ground_truth_class}</span></span>
                               <div className="flex gap-2 opacity-0 group-hover:opacity-100 transition-opacity">
                                  <button className="p-1 bg-[#1a1a1a] rounded text-xs hover:bg-[#313030]" title="View in Embedding Explorer">🔭</button>
                                  <button className="p-1 bg-[#1a1a1a] rounded text-xs hover:bg-[#313030]" title="Find Similar Images">🔍</button>
                               </div>
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                  
                  {poorLocalization.length > 0 && (
                    <div>
                      <h3 className="text-sm font-geist font-medium text-gray-400 mb-4 border-b border-[#313030] pb-2 uppercase tracking-wider">Poor Localization (IoU {'<'} 0.5)</h3>
                      <div className="grid grid-cols-2 gap-4">
                        {poorLocalization.map((f, i) => (
                          <div key={i} className="bg-[#131313] border border-[#313030] rounded p-3 hover:border-[#a6c8ff] transition-colors cursor-pointer group">
                            <div className="aspect-video bg-[#0A0A0A] rounded mb-3 flex items-center justify-center border border-[#1c1b1b] relative overflow-hidden text-xs text-gray-600">
                               [Image: {f.image_id}]
                            </div>
                            <div className="flex justify-between items-center">
                               <span className="text-xs text-gray-400">IoU: <span className="text-[#ffb875] font-medium">{f.iou?.toFixed(2)}</span></span>
                               <div className="flex gap-2 opacity-0 group-hover:opacity-100 transition-opacity">
                                  <button className="p-1 bg-[#1a1a1a] rounded text-xs hover:bg-[#313030]" title="View in Embedding Explorer">🔭</button>
                                  <button className="p-1 bg-[#1a1a1a] rounded text-xs hover:bg-[#313030]" title="Find Similar Images">🔍</button>
                               </div>
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
