"use client";

import React, { useState } from "react";

export default function BenchmarkPage() {
  const [comparing, setComparing] = useState(false);
  
  return (
    <div className="flex flex-col min-h-screen bg-[#0A0A0A] text-[#e5e2e1] p-8 font-inter">
      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-3xl font-geist font-semibold tracking-tight">Benchmark Lab</h1>
          <p className="text-sm text-gray-400 mt-1">Controlled Computer Vision Benchmarking Platform</p>
        </div>
        <button 
          onClick={() => setComparing(true)}
          className="px-4 py-2 bg-[#005fb0] hover:bg-[#3192fc] text-white rounded-md font-medium transition-colors"
        >
          Run Benchmark
        </button>
      </div>

      {/* Experimental Configuration */}
      <div className="bg-[#111111] border border-[#313030] rounded-lg p-5 mb-8 flex justify-between">
        <div>
          <h3 className="text-sm font-geist font-semibold text-gray-400 mb-2 uppercase tracking-wider">Dataset</h3>
          <div className="font-mono text-sm text-[#a6c8ff]">safety_v2</div>
          <div className="text-xs text-gray-500 mt-1">Prep: #12, Split: test</div>
        </div>
        <div>
          <h3 className="text-sm font-geist font-semibold text-gray-400 mb-2 uppercase tracking-wider">Models</h3>
          <div className="flex flex-col gap-1">
             <label className="flex items-center text-sm"><input type="checkbox" checked readOnly className="mr-2" /> YOLO11s (CNN)</label>
             <label className="flex items-center text-sm"><input type="checkbox" checked readOnly className="mr-2" /> RT-DETR-L (ViT)</label>
          </div>
        </div>
        <div>
          <h3 className="text-sm font-geist font-semibold text-gray-400 mb-2 uppercase tracking-wider">Configuration</h3>
          <div className="grid grid-cols-2 gap-x-4 gap-y-1 text-xs text-gray-300">
             <span>Epochs: <span className="text-white font-mono">50</span></span>
             <span>Batch: <span className="text-white font-mono">16</span></span>
             <span>Image Size: <span className="text-white font-mono">640px</span></span>
             <span>Device: <span className="text-white font-mono">Colab T4</span></span>
             <span>Seed: <span className="text-white font-mono">42</span></span>
          </div>
        </div>
      </div>

      <div className="bg-[#111111] border border-[#313030] rounded-lg overflow-hidden mb-8">
        <div className="px-5 py-4 border-b border-[#313030] bg-[#1a1a1a]">
          <h2 className="font-geist font-semibold">Results Comparison</h2>
        </div>
        <div className="p-0">
          <table className="w-full text-sm text-left">
            <thead className="text-xs uppercase bg-[#131313] text-gray-400 font-geist">
              <tr>
                <th className="px-5 py-3">Metric</th>
                <th className="px-5 py-3">visionforge_yolo11s (CNN)</th>
                <th className="px-5 py-3">visionforge_rtdetr-l (ViT)</th>
                <th className="px-5 py-3">Delta</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-[#313030]">
              <tr className="hover:bg-[#1a1a1a]">
                <td className="px-5 py-4 font-medium text-gray-300">mAP@50</td>
                <td className="px-5 py-4 font-mono text-white">84.5%</td>
                <td className="px-5 py-4 font-mono text-[#51df8e] font-semibold">{comparing ? "86.2%" : "---"}</td>
                <td className="px-5 py-4 font-mono text-[#51df8e]">{comparing ? "+1.7%" : "---"}</td>
              </tr>
              <tr className="hover:bg-[#1a1a1a]">
                <td className="px-5 py-4 font-medium text-gray-300">Precision</td>
                <td className="px-5 py-4 font-mono text-white">86.4%</td>
                <td className="px-5 py-4 font-mono text-white">{comparing ? "85.8%" : "---"}</td>
                <td className="px-5 py-4 font-mono text-[#ffb4ab]">{comparing ? "-0.6%" : "---"}</td>
              </tr>
              <tr className="hover:bg-[#1a1a1a]">
                <td className="px-5 py-4 font-medium text-gray-300">Recall</td>
                <td className="px-5 py-4 font-mono text-white">81.2%</td>
                <td className="px-5 py-4 font-mono text-[#51df8e] font-semibold">{comparing ? "84.3%" : "---"}</td>
                <td className="px-5 py-4 font-mono text-[#51df8e]">{comparing ? "+3.1%" : "---"}</td>
              </tr>
              <tr className="hover:bg-[#1a1a1a]">
                <td className="px-5 py-4 font-medium text-gray-300">Latency (T4)</td>
                <td className="px-5 py-4 font-mono text-[#51df8e] font-semibold">12.5 ms</td>
                <td className="px-5 py-4 font-mono text-[#ffb4ab]">{comparing ? "24.1 ms" : "---"}</td>
                <td className="px-5 py-4 font-mono text-[#ffb4ab]">{comparing ? "+11.6 ms" : "---"}</td>
              </tr>
              <tr className="hover:bg-[#1a1a1a]">
                <td className="px-5 py-4 font-medium text-gray-300">Throughput</td>
                <td className="px-5 py-4 font-mono text-[#51df8e] font-semibold">80 FPS</td>
                <td className="px-5 py-4 font-mono text-[#ffb4ab]">{comparing ? "41 FPS" : "---"}</td>
                <td className="px-5 py-4 font-mono text-[#ffb4ab]">{comparing ? "-39 FPS" : "---"}</td>
              </tr>
              <tr className="hover:bg-[#1a1a1a]">
                <td className="px-5 py-4 font-medium text-gray-300">Parameters</td>
                <td className="px-5 py-4 font-mono text-[#51df8e] font-semibold">11.1 M</td>
                <td className="px-5 py-4 font-mono text-[#ffb4ab]">{comparing ? "32.0 M" : "---"}</td>
                <td className="px-5 py-4 font-mono text-[#ffb4ab]">{comparing ? "+20.9 M" : "---"}</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
      
      {comparing && (
        <div className="grid grid-cols-2 gap-8 mb-8">
          <div className="bg-[#111111] border border-[#313030] rounded-lg p-5">
             <h3 className="text-sm font-geist font-semibold mb-3">Accuracy / Efficiency Trade-off</h3>
             <p className="text-sm text-gray-400 mb-2">
                <span className="text-[#a6c8ff] mr-2 font-semibold">Observation:</span>
                RT-DETR-L provides a <span className="text-[#51df8e]">1.7% increase</span> in mAP@50, driven primarily by much higher Recall on smaller objects (like background safety goggles).
             </p>
             <p className="text-sm text-gray-400">
                <span className="text-[#ffb875] mr-2 font-semibold">Cost:</span>
                This comes at the cost of almost <span className="text-[#ffb4ab]">double the inference latency</span> (24.1ms vs 12.5ms), dropping throughput from 80 FPS to 41 FPS.
             </p>
          </div>
          
          <div className="bg-[#111111] border border-[#313030] rounded-lg p-5">
             <h3 className="text-sm font-geist font-semibold mb-3">Qualitative Failure Difference</h3>
             <p className="text-sm text-gray-400 mb-2">
                <span className="text-white mr-2">CNN Bias:</span>
                YOLO11s struggles with heavily occluded overlapping classes (e.g. helmet on head), resulting in misclassifications.
             </p>
             <p className="text-sm text-gray-400">
                <span className="text-white mr-2">Transformer Advantage:</span>
                RT-DETR's global attention mechanism allows it to capture context better, reducing false negatives in crowded scenes.
             </p>
          </div>
        </div>
      )}

      {comparing && (
        <div className="bg-[#111111] border border-[#313030] rounded-lg overflow-hidden flex flex-col">
          <div className="px-5 py-4 border-b border-[#313030] bg-[#1a1a1a] flex items-center justify-between">
            <h2 className="font-geist font-semibold">Visual Comparison Gallery</h2>
          </div>
          
          <div className="p-5 flex-1">
             <h3 className="text-sm font-geist font-medium text-gray-400 mb-4 border-b border-[#313030] pb-2 uppercase tracking-wider">Example: Crowded Scene Occlusion</h3>
             <div className="grid grid-cols-3 gap-4">
                <div className="bg-[#131313] border border-[#313030] rounded p-3">
                   <div className="text-center text-xs text-gray-400 mb-2">Ground Truth</div>
                   <div className="aspect-video bg-[#0A0A0A] rounded flex items-center justify-center border border-[#1c1b1b] relative overflow-hidden">
                      <div className="absolute inset-0 border border-green-500 m-4 flex items-start p-1 text-[8px] text-green-500 font-mono">helmet</div>
                      <div className="absolute inset-2 border border-blue-500 m-4 flex items-start p-1 text-[8px] text-blue-500 font-mono mt-6">head</div>
                   </div>
                </div>
                
                <div className="bg-[#131313] border border-[#ffb4ab]/30 rounded p-3">
                   <div className="text-center text-xs text-gray-400 mb-2 font-mono">YOLO11s Predictions</div>
                   <div className="aspect-video bg-[#0A0A0A] rounded flex items-center justify-center border border-[#1c1b1b] relative overflow-hidden">
                      {/* YOLO missed the head behind the helmet */}
                      <div className="absolute inset-0 border border-green-500 m-4 flex items-start p-1 text-[8px] text-green-500 font-mono bg-green-500/10">helmet 0.92</div>
                   </div>
                   <div className="text-center text-xs text-[#ffb4ab] mt-2 font-medium">FALSE NEGATIVE (Missed Head)</div>
                </div>

                <div className="bg-[#131313] border border-[#51df8e]/30 rounded p-3">
                   <div className="text-center text-xs text-gray-400 mb-2 font-mono">RT-DETR-L Predictions</div>
                   <div className="aspect-video bg-[#0A0A0A] rounded flex items-center justify-center border border-[#1c1b1b] relative overflow-hidden">
                      {/* RT-DETR successfully detected both */}
                      <div className="absolute inset-0 border border-green-500 m-4 flex items-start p-1 text-[8px] text-green-500 font-mono bg-green-500/10">helmet 0.95</div>
                      <div className="absolute inset-2 border border-blue-500 m-4 flex items-start p-1 text-[8px] text-blue-500 font-mono mt-6 bg-blue-500/10">head 0.81</div>
                   </div>
                   <div className="text-center text-xs text-[#51df8e] mt-2 font-medium">SUCCESS</div>
                </div>
             </div>
          </div>
        </div>
      )}
    </div>
  );
}
